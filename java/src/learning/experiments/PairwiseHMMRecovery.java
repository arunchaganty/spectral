package learning.experiments;

import fig.basic.*;
import fig.exec.Execution;
import learning.linalg.FullTensor;
import learning.linalg.MatrixFactory;
import learning.linalg.MatrixOps;
import learning.linalg.RandomFactory;
import learning.models.BasicParams;
import learning.models.HiddenMarkovModel;
import learning.spectral.TensorMethod;
import learning.spectral.applications.ParameterRecovery;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

import java.util.Arrays;
import java.util.Random;

import static fig.basic.LogInfo.*;
import static learning.common.Utils.doGradientCheck;
import static learning.common.Utils.outputList;

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

  @Option(gloss="init")
  public Random initRandom = new Random(42);
  @Option(gloss="init")
  public double initRandomNoise = 1.0;
  @Option(gloss="Start at the exact solution?")
  public boolean initExact = false;


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

  double lhoodT(int K, int D, int[][] X, double[][] O, double[][] P) {
    double count = 0.;
    double lhood = 0.;
    for(int[] x : X) {
      for(int i = 0; i < x.length-1;i++) {
        count++;
        int x1 = x[i+0];
        int x2 = x[i+1];

        double z = 0.;
        for(int h1 = 0; h1 < K; h1++)
          for(int h2 = 0; h2 < K; h2++)
            z += P[h1][h2] * O[h1][x1] * O[h2][x2];
        // Update
        lhood += (Math.log(z) - lhood)/count;
      }
    }

    return lhood;
  }

  public double[][] getHessian(HiddenMarkovModel model, int[][] X, double[][] O, double[][] P) {
    int K = model.getStateCount();
    int D = model.getEmissionCount();

    double[][] H = new double[K*K][K*K];

    int count = 0;
    for(int[] x : X) {
      for(int offset = 0; offset < x.length-1; offset++) {
        count++;
        int x1 = x[offset];
        int x2 = x[offset+1];

        double z = 0.;
        for(int h1 = 0; h1 < K; h1++) {
          for(int h2 = 0; h2 < K; h2++) {
            z += P[h1][h2] * O[h1][x1] * O[h2][x2];
          }
        }

        for(int i = 0; i < K * K; i++) {
          int h1 = i / K;
          int h2 = i % K;
          for(int j = 0; j < K * K; j++) {
            int h1_ = j / K;
            int h2_ = j % K;

            // Add O_z
            H[i][j] += (-(O[h1][x1] * O[h2][x2]/z) * (O[h1_][x1] * O[h2_][x2]/z) - H[i][j])/count;
          }
        }
      }
    }

    return H;
  }
  public boolean checkIdentifiability(double[][] hessian) {
    assert(hessian.length == hessian[0].length);
    log(MatrixOps.svd(new SimpleMatrix(hessian)).getValue1());
    int rank = MatrixOps.rank(new SimpleMatrix(hessian));
    logs("Problem has rank %d vs %d.", rank, hessian.length);
    return rank == hessian.length;
  }


  public double[][] recoverTviaEM(HiddenMarkovModel model, int[][] X, double[][] O) {

    LogInfo.begin_track("recover-T-em");
    int K = model.getStateCount();
    int D = model.getEmissionCount();

    // Initialization
    double[][] P = RandomFactory.rand_(initRandom, K,K);
    MatrixOps.abs(P);
    MatrixOps.scale(P, 1./MatrixOps.sum(P));
    {
      double[][] T = model.params.T;
      double[][] P_ = new double[K][K];
      for(int h1 = 0; h1 < K; h1++) {
        for(int h2 = 0; h2 < K; h2++) {
          P_[h1][h2] = model.params.pi[h1] * T[h1][h2];
        }
      }
      LogInfo.log("Is identifable?: " + checkIdentifiability(getHessian(model, X, O, P_)));
      if(initExact) {
        for(int h1 = 0; h1 < K; h1++) {
          System.arraycopy(P_[h1], 0, P[h1], 0, K);
        }
      }
    }
    assert( MatrixOps.equal(MatrixOps.sum(P), 1.0) );


    double[][] P_ = new double[K][K];
    boolean done = false;
    double count = 0;

    double lhood_old = Double.NEGATIVE_INFINITY;
    for(int iter = 0; iter < 1000 && !done; iter++){
      // Compute marginals
      double lhood = 0.;
      for(int[] x : X) {
        for(int i = 0; i < x.length-1;i++) {
          count++;

          int x1 = x[i+0];
          int x2 = x[i+1];

          double z = 0.;
          for(int h1 = 0; h1 < K; h1++)
            for(int h2 = 0; h2 < K; h2++)
              z += P[h1][h2] * O[h1][x1] * O[h2][x2];

          // Update
          lhood += (Math.log(z) - lhood)/count;
          for(int h1 = 0; h1 < K; h1++)
            for(int h2 = 0; h2 < K; h2++)
              P_[h1][h2] += ((P[h1][h2] * O[h1][x1] * O[h2][x2]) / z - P_[h1][h2]) / count;
        }
      }

      // Update
      double diff = 0.;
      for(int h1 = 0; h1 < K; h1++) {
        for(int h2 = 0; h2 < K; h2++) {
          diff += Math.abs(P[h1][h2] - P_[h1][h2]);
          P[h1][h2] = P_[h1][h2];
        }
      }
      assert( MatrixOps.equal(MatrixOps.sum(P), 1.0 ));

      assert(lhood > lhood_old);
      lhood_old = lhood;

      LogInfo.log(outputList(
              "iter", iter,
              "lhood", lhood,
              "diff", diff
      ));

      done = diff < 1e-3;
    }

    logs("Likelihood-T: %.3f", lhood_old);

    // -- Now compute T.
    double[][] T = new double[K][K];
    for(int h1 = 0; h1 < K; h1++) {
      double Z = 0.;
      for(int h2 = 0; h2 < K; h2++) {
        Z += P[h1][h2];
      }
      for(int h2 = 0; h2 < K; h2++) {
        T[h1][h2] = P[h1][h2] / Z;
      }
    }

    end_track("recover-T-em");
    return T;
  }

  public double[][] recoverT(HiddenMarkovModel model, int[][] X, double[][] O) {
    LogInfo.begin_track("recover-T");
    int K = model.getStateCount();
    int D = model.getEmissionCount();
    // Next minimize the pairwise objective
    PairwiseLikelihood state = new PairwiseLikelihood(K, D, X, O);
    // Initialize with exact:
    if(initExact) {
      double[] P = state.point();
      double[][] T = model.params.T;
      for(int h1 = 0; h1 < K; h1++) {
        for(int h2 = 0; h2 < K; h2++) {
          P[state.index(h1,h2)] = model.params.pi[h1] * T[h1][h2];
        }
      }
      assert( MatrixOps.equal(MatrixOps.sum(P), 1.0));
      assert( MatrixOps.equal(MatrixOps.sum(state.point()), 1.0));
    }

//    Maximizer max = newMaximizer();
    Maximizer max = new MultiplicativeUpdates(1.0);

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

  public double[] recoverPi(HiddenMarkovModel model, int[][] X, double[][] O) {
    LogInfo.begin_track("recover-pi");
    int K = model.getStateCount();
    int D = model.getEmissionCount();

    // Initialization
    double[] pi = new double[K];
    Arrays.fill(pi, 1./K);
    if(initExact) {
      System.arraycopy(model.params.pi, 0, pi, 0, K);
    }
    assert( MatrixOps.equal(MatrixOps.sum(pi), 1.0) );

    double[] pi_ = new double[K];
    boolean done = false;
    double count = 0;
    double lhood_old = Double.NEGATIVE_INFINITY;
    for(int iter = 0; iter < 100 && !done; iter++){
      // Compute marginals
      double lhood = 0.;
      for(int[] x : X) {
        int x1 = x[0];
        double Z = 0.;
        for(int h1 = 0; h1 < K; h1++)
          Z += pi[h1] * O[h1][x1];

        count++;

        lhood += (Math.log(Z) - lhood)/count;
        for(int h1 = 0; h1 < K; h1++)
          pi_[h1] += ((pi[h1] * O[h1][x1] / Z) - pi_[h1]) / count;
      }

      // Update
      double diff = 0.;
      for(int h1 = 0; h1 < K; h1++) {
        diff += Math.abs(pi[h1] - pi_[h1]);
        pi[h1] = pi_[h1];
      }
      assert( MatrixOps.equal(MatrixOps.sum(pi), 1.0 ));

      assert(lhood > lhood_old);
      lhood_old = lhood;

      LogInfo.log(outputList(
              "iter", iter,
              "lhood", lhood,
              "diff", diff
      ));

      done = diff < 1e-3;
    }

    LogInfo.log(Fmt.D(pi));

    end_track("recover-pi");
    return pi;
  }

  public HiddenMarkovModel spectralConvexRecovery(HiddenMarkovModel model, int[][] X) {
    // -- Compose moments
    int K = model.getStateCount();
    int D = model.getEmissionCount();

    // Now factorize!
    begin_track("recover-O");
    double[][] O;
    SimpleMatrix O_;
    if(true || initExact) {
      O = model.params.O.clone();
      O_ = new SimpleMatrix(O).transpose();
    } else {
      FullTensor momentsO = computeMomentsO(D, X);
      TensorMethod tensorMethod = new TensorMethod();
      Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, SimpleMatrix> params = tensorMethod.recoverParametersAsymmetric(K, momentsO);

      O_ = params.getValue3();
      // Project onto simplex
  //    SimpleMatrix OT = params.getValue2();
      O_ = MatrixOps.projectOntoSimplex(O_, smoothMeasurements);
      O = MatrixFactory.toArray(O_.transpose());
      assert O.length == K;
      assert O[0].length == D;
    }
    // Compare O with model?
    O_ = MatrixOps.alignMatrix( O_, model.getO(), true );
    log(outputList(
            "O-error", MatrixOps.diff(O_, model.getO()),
            "\nO^", O_,
            "\nO*", model.getO()
    ));

    end_track("recover-O");

    // Compute pi with maximum likelihood.
    double[] pi = recoverPi(model, X, O);
    SimpleMatrix pi_ = MatrixFactory.fromVector(pi);
    pi_ = MatrixOps.alignMatrix( pi_, model.getPi().transpose(), false );
    log(outputList(
            "pi-error", MatrixOps.diff(pi_, model.getPi().transpose()),
            "\npi^", pi_,
            "\npi*", model.getPi().transpose()
    ));

    double[][] T = recoverTviaEM(model, X, O);
    SimpleMatrix T_ = new SimpleMatrix(T).transpose();
    T_ = MatrixOps.alignMatrix( T_, model.getT(), true );
    log(outputList(
            "T-error", MatrixOps.diff(T_, model.getT()),
            "\nT^", T_,
            "\nT*", model.getT()
    ));

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
    int[][] X = model.sample(options.genRandom, (int)N, L);

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
