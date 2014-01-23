package learning.experiments;

import Jama.Matrix;
import fig.basic.*;
import fig.exec.Execution;
import learning.linalg.FullTensor;
import learning.linalg.MatrixFactory;
import learning.linalg.MatrixOps;
import learning.models.BasicParams;
import learning.models.HiddenMarkovModel;
import learning.spectral.TensorMethod;
import learning.spectral.applications.ParameterRecovery;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

import java.util.Arrays;
import java.util.Random;

import static fig.basic.LogInfo.begin_track;
import static fig.basic.LogInfo.end_track;
import static fig.basic.LogInfo.log;
import static learning.common.Utils.doGradientCheck;
import static learning.common.Utils.outputList;
import static learning.common.Utils.writeString;

/**
 * Recover a HMM from pairwise factors
 * - First solve for observations
 * - Then solve the convex likelihood.
 */
public class PairwiseHMMRecovery implements  Runnable {
  @Option(gloss="Data used")
  public double N = 1e2;
  @Option(gloss="Sequence length")
  public int L = 5;

  @OptionSet(name = "genOpts")
  public HiddenMarkovModel.GenerationOptions options = new HiddenMarkovModel.GenerationOptions();

  @Option(gloss="generation")
  public Random genRand = new Random(42);

  @Option(gloss="init")
  public Random initRandom = new Random(42);
  @Option(gloss="init")
  public double initRandomNoise = 1.0;

  @Option(gloss="iterations to run EM")
  public int iters = 100;

  @Option(gloss="Run EM?")
  public boolean runEM = true;

  @Option(gloss="How much to smooth")
  public double smoothMeasurements = 1e-2;

  @Option(gloss="Type of optimization to use") public boolean useLBFGS = false;
  @OptionSet(name="lbfgs") public LBFGSMaximizer.Options lbfgs = new LBFGSMaximizer.Options();
  @OptionSet(name="backtrack") public BacktrackingLineSearch.Options backtrack = new BacktrackingLineSearch.Options();

  Maximizer newMaximizer() {
    if (useLBFGS) return new LBFGSMaximizer(backtrack, lbfgs);
    return new GradientMaximizer(backtrack);
  }

  enum RunMode {
    EM,
    EMGoodStart,
    SpectralInitialization,
    SpectralConvex
  }
  @Option
  public RunMode mode = RunMode.EM;

  public FullTensor computeMomentsO(int D, int[][] X) {
    double[][][] moments = new double[D][D][D];
    int[][][] counts = new int[D][D][D];
    int it = 0;
    for(int [] x : X) {
      // Sliding window
      for(int i = 0; i < x.length - 2; i++ ) {
        it++;
        int x1 = x[i+0];
        int x2 = x[i+2];
        int x3 = x[i+1]; // Because we want the middle entry to be M3.
        // Update the moments -> All the values from counts to now (it) were zero
        int updates = it - counts[x1][x2][x3];
        moments[x1][x2][x3] += (1. - updates * moments[x1][x2][x3]) / it;
        counts[x1][x2][x3] = it;
      }
    }
    // Final counting of zeros
    for(int x1 = 0; x1 < D; x1++) {
      for(int x2 = 0; x2 < D; x2++) {
        for(int x3 = 0; x3 < D; x3++) {
          int updates = it - counts[x1][x2][x3];
          moments[x1][x2][x3] += (-updates * moments[x1][x2][x3]) / it;
          counts[x1][x2][x3] = it;
        }
      }
    }
    // Some checksums
    double sum = MatrixOps.sum(moments);
    assert MatrixOps.equal(sum, 1.0);

    return new FullTensor(moments);
  }

  static class PairwiseLikelihood implements Maximizer.FunctionState {
    final int K;
    final int D;
    final int[][] X;
    final double[] weights;
    final double[] gradient;
    final double[][] O;

    int index(int h, int h_) {
      return h * K + h_;
    }

    public PairwiseLikelihood(int K, int D, int[][] X, double[][] O) {
      this.K = K;
      this.D = D;
      this.X = X;
      this.O = O;

      weights = new double[K*K];
      // Initialize with all ones (feasible)
      Arrays.fill(weights, 1./(K*K));
      gradient = new double[K*K];
    }

    @Override
    public double[] point() {
      return weights;
    }

    @Override
    public double value() {
      // Pairwise Likelihood
      double lhood = 0.;
      int count = 0;
      for(int[] x : X) {
        for(int i = 0; i < x.length-1; i++) {
          int x1 = x[i+0];
          int x2 = x[i+1];

          count++;
          double z = 0.;
          for(int h1 = 0; h1 < K; h1++) {
            for(int h2 = 0; h2 < K; h2++) {
              double prob = weights[index(h1,h2)] * O[h1][x1] * O[h2][x2];
              z += prob;
            }
          }
          lhood += (Math.log(z) - lhood)/count;
        }
      }

      return lhood;
    }

    @Override
    public double[] gradient() {
      Arrays.fill(gradient, 0.);
      int count = 0;
      for(int[] x : X) {
        for(int i = 0; i < x.length-1; i++) {
          int x1 = x[i+0];
          int x2 = x[i+1];

          double Z = 0.;
          for(int h1 = 0; h1 < K; h1++)
            for(int h2 = 0; h2 < K; h2++)
              Z += weights[index(h1,h2)] * O[h1][x1] * O[h2][x2];

          count++;
          for(int h1 = 0; h1 < K; h1++)
            for(int h2 = 0; h2 < K; h2++)
              gradient[index(h1,h2)] += (O[h1][x1] * O[h2][x2] / Z - gradient[index(h1,h2)])/count; // Same sort of running average.
        }
      }

//      MatrixOps.scale(gradient, -1.0);

      return gradient;
    }

    @Override
    public void invalidate() {

    }
  }

  class MultiplicativeUpdates extends Maximizer {
    double stepSize;
    int steps = 0;

    public MultiplicativeUpdates(double stepSize) {
      this.stepSize = stepSize;
    }
    public MultiplicativeUpdates() {
      this(1.0);
    }

    @Override
    public boolean takeStep(FunctionState func) {
      // w_{t+1} \propto w_t e^{\eta z_t}
      double[] point = func.point();
      double[] gradient = func.gradient();

      for(int i = 0; i < point.length; i++) {
        point[i] *= Math.exp(stepSize * gradient[i]);
      }
      double Z = MatrixOps.sum(point);
      MatrixOps.scale(point, 1./Z);

      // Anneal the step size. (1/sqrt(1+t))
      stepSize = stepSize * Math.sqrt((1+steps)/(2+steps));
      steps++;
      return false;
    }
  }

  public double[][] recoverT(HiddenMarkovModel model, int[][] X, double[][] O) {
    LogInfo.begin_track("recover-T");
    int K = model.getStateCount();
    int D = model.getEmissionCount();
    // Next minimize the pairwise objective
    PairwiseLikelihood state = new PairwiseLikelihood(K, D, X, O);
    // Initialize with exact:
    {
      double[][] T_ = model.params.T;
      for(int h = 0; h < K; h++) {
        for(int h_ = 0; h_ < K; h_++) {
          state.point()[state.index(h,h_)] = model.params.pi[h_] * T_[h][h_];
        }
      }
      MatrixOps.normalize(state.point());
    }

    Maximizer max = newMaximizer();
//    Maximizer max = new MultiplicativeUpdates(1.0);

    int numIters = 100;
    boolean done = false;
    double oldObjective = Double.NEGATIVE_INFINITY;
    for (int iter = 0; iter < numIters && !done; iter++) {
      doGradientCheck(state);
      // Logging stuff
      LogInfo.log(outputList(
              "iter", iter,
              "objective", state.value(),
              "pointNorm", MatrixOps.norm(state.point()),
              "gradientNorm", MatrixOps.norm(state.gradient()),
              "pointSum", MatrixOps.sum(state.point())
      ));
      LogInfo.log(Fmt.D(state.point()));
      LogInfo.log(Fmt.D(state.gradient()));
      double objective = state.value();
      if(objective < oldObjective) break;
      if(Math.abs(objective - oldObjective) < 1e-3) break;
      assert( objective > oldObjective );
      oldObjective = objective;

      assert MatrixOps.equal(MatrixOps.sum(state.point()), 1.0);
      done = max.takeStep(state);
      // Project back onto simplex
      MatrixOps.projectOntoSimplex(state.point());
    }

    LogInfo.log(outputList(
            "done", done,
            "objective", state.value(),
            "pointNorm", MatrixOps.norm(state.point()),
            "gradientNorm", MatrixOps.norm(state.gradient()),
            "pointSum", MatrixOps.sum(state.point())
    ));

    // -- Now compute T.
    double[] P = state.point();
    log(Fmt.D(P));

    double[][] T = new double[K][K];
    for(int h = 0; h < K; h++) {
      double Z = 0.;
      for(int h_ = 0; h_ < K; h_++) {
        Z += P[state.index(h,h_)];
      }
      for(int h_ = 0; h_ < K; h_++) {
        T[h][h_] = P[state.index(h,h_)] / Z;
      }
    }
    end_track("recover-T");
    return T;
  }

  public HiddenMarkovModel spectralConvexRecovery(HiddenMarkovModel model, int[][] X) {
    // -- Compose moments
    int K = model.getStateCount();
    int D = model.getEmissionCount();

    // Now factorize!
    begin_track("recover-O");
//    FullTensor momentsO = computeMomentsO(D, X);
//    TensorMethod tensorMethod = new TensorMethod();
//    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, SimpleMatrix> params = tensorMethod.recoverParametersAsymmetric(K, momentsO);
//
//    SimpleMatrix O_ = params.getValue3();
//    // Project onto simplex
////    SimpleMatrix OT = params.getValue2();
//    O_ = MatrixOps.projectOntoSimplex(O_, smoothMeasurements);
//
//    double[][] O = MatrixFactory.toArray(O_.transpose());
//    assert O.length == K;
//    assert O[0].length == D;

    double[][] O = model.params.O;
    {
      SimpleMatrix O_ = new SimpleMatrix(O).transpose();
      // Compare O with model?
      log(O_);
      log(model.getO());
      log(outputList("O-error", MatrixOps.diff(O_, model.getO())));
      O_ = MatrixOps.alignMatrix( O_, model.getO(), true );
      log(outputList("O-error", MatrixOps.diff(O_, model.getO())));
    }

    end_track("recover-O");

    begin_track("recover-pi");
    // Compute pi with maximum likelihood.
    double[] pi = new double[K];
    int count = 0;
    for(int[] x : X) {
      int x1 = x[0];
      double Z = 0.;
      for(int h = 0; h < K; h++) {
        Z += O[h][x1];
      }

      // Moving average
      count++;
      for(int h = 0; h < K; h++) {
        pi[h] += (O[h][x1] / Z - pi[h])/count;
      }
    }
    SimpleMatrix pi_ = MatrixFactory.fromVector(pi);
    log(pi_);
    log(model.getPi().transpose());
    log(outputList("pi-error", MatrixOps.diff(pi_, model.getPi().transpose())));
    pi_ = MatrixOps.alignMatrix( pi_, model.getPi().transpose(), true );
    log(outputList("pi-error", MatrixOps.diff(pi_, model.getPi().transpose())));
    end_track("recover-pi");

    double[][] T = recoverT(model, X, O);
    SimpleMatrix T_ = new SimpleMatrix(T).transpose();
    log(T_);
    log(model.getT());
    log(outputList("T-error", MatrixOps.diff(T_, model.getT())));
    T_ = MatrixOps.alignMatrix( T_, model.getT(), true );
    log(outputList("T-error", MatrixOps.diff(T_, model.getT())));
    log(T_);

    return new HiddenMarkovModel(new HiddenMarkovModel.Params(pi, T, O));
  }

  @Override
  public void run() {
    int D = options.emissionCount;
    int K = options.stateCount;

    HiddenMarkovModel model = HiddenMarkovModel.generate(options);
    log(model.getPi());
    log(model.getT());
    log(model.getO());
    // Get data
    int[][] X = model.sample(genRand, (int)N, L);

    BasicParams params = model.toParams();

    Execution.putOutput("true-likelihood", model.likelihood(X));
    Execution.putOutput("true-paramsError", model.toParams().computeDiff(params, null));
    Execution.putOutput("true-pi", model.getPi());
    Execution.putOutput("true-T", model.getT());
    Execution.putOutput("true-O", model.getO());

    log(outputList(
            "true-likelihood", model.likelihood(X),
            "true-paramsError", model.toParams().computeDiff(params, null)
    ));

    // Process via EM or Spectral
    HiddenMarkovModel model_;
    switch(mode) {
      case EM: {
        model_ = new HiddenMarkovModel(
                HiddenMarkovModel.Params.uniformWithNoise(initRandom, K, D, initRandomNoise));
        runEM = true; // Regardless of what you said before.
      } break;
      case EMGoodStart: {
        model_ = new HiddenMarkovModel( model.getParams().clone() );
        runEM = true; // Regardless of what you said before.
      } break;
      case SpectralInitialization: {
        model_ = ParameterRecovery.recoverHMM(K, (int)N, model, smoothMeasurements);
        // Align with true parameters.
        SimpleMatrix O = model_.getO();
        SimpleMatrix O_ = MatrixOps.alignMatrix(O, model.getO(), true);
        log(outputList("O-error", MatrixOps.diff(O_, model.getO())));
      } break;
      case SpectralConvex: {
        model_ = spectralConvexRecovery(model, X);
      } break;
      default:
        throw new RuntimeException("Not implemented");
    }

    Execution.putOutput("initial-likelihood", model_.likelihood(X));
    Execution.putOutput("initial-paramsError", model_.toParams().computeDiff(params, null));
    Execution.putOutput("initial-pi", model_.getPi());
    Execution.putOutput("initial-T", model_.getT());
    Execution.putOutput("initial-O", model_.getO());

    if(runEM) {
      double lhood_ = Double.NEGATIVE_INFINITY;
      for(int i = 0; i < iters; i++) {
        // Print error per iteration.
        double lhood = model_.likelihood(X);
        log(outputList(
                "iter", i,
                "likelihood", lhood,
                "paramsError", params.computeDiff(model_.toParams(), null)
        ));

        assert lhood > lhood_;
        lhood_ = lhood;
        model_.baumWelchStep(X);
      }
      log(outputList(
              "likelihood", model_.likelihood(X),
              "paramsError", params.computeDiff(model_.toParams(), null)
      ));
    }

    Execution.putOutput("final-likelihood", model_.likelihood(X));
    Execution.putOutput("final-paramsError", params.computeDiff(model_.toParams(), null));
    Execution.putOutput("final-pi", model_.getPi());
    Execution.putOutput("final-T", model_.getT());
    Execution.putOutput("final-O", model_.getO());
  }

  public static void main(String[] args) {
    Execution.run(args, new PairwiseHMMRecovery());
  }
}
