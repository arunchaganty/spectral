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
  @Option(gloss="Start at the exact solution?")
  public boolean initExactO = false;


  @Option(gloss="iterations to run EM")
  public int iters = 100;

  @Option(gloss="Run EM?")
  public boolean runEM = false;

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

  public double[][] computeLikelihoodHessian(HiddenMarkovModel model, int[][] X) {
    final double eps = 1e-6;
    double[] weights = model.params.toVector();
    double[] original = weights.clone();
    double[][] H = new double[weights.length][weights.length];

    double lhood = model.likelihood(X);
    for(int i = 0; i < weights.length; i++) {
      for(int j = 0; j < weights.length; j++) {
        weights[i] += eps;
        weights[j] += eps;
        model.params.updateFromVector(weights);
        double lhoodPlusPlus = model.likelihood(X);

        weights[j] -= 2*eps;
        model.params.updateFromVector(weights);
        double lhoodPlusMinus = model.likelihood(X);

        weights[i] -= 2*eps;
        model.params.updateFromVector(weights);
        double lhoodMinusMinus = model.likelihood(X);

        weights[j] += 2*eps;
        model.params.updateFromVector(weights);
        double lhoodMinusPlus = model.likelihood(X);

        weights[i] += eps;
        weights[j] -= eps;

        H[i][j] = ((lhoodPlusPlus - lhoodMinusPlus)/(2*eps) + (lhoodMinusMinus  - lhoodPlusMinus)/(2*eps))/(2*eps);
      }
    }
    model.params.updateFromVector(original);

    return H;
  }

  public boolean fullIdentifiabilityReport(HiddenMarkovModel model, int[][] X) {
    int K = model.getStateCount();
    int D = model.getEmissionCount();

    double[][] H = computeLikelihoodHessian(model,X);

    assert(H.length == H[0].length);
    SimpleMatrix sigma = MatrixOps.svd(new SimpleMatrix(H)).getValue1().extractDiag().transpose();
    log(sigma);
    int rank = MatrixOps.rank(new SimpleMatrix(H));
    logs("Problem has rank %d vs %d.", rank, H.length);
    Execution.putOutput("full-sigmak", sigma.get(H.length-1));
    Execution.putOutput("full-K", sigma.get(0,0) / sigma.get(H.length-1));
    return rank == H.length;
  }

  public double[][] getPiecewiseParameters(HiddenMarkovModel model) {
    int K = model.getStateCount();
    double[][] T = model.params.T;
    double[][] P = new double[K][K];

    // Preprocess pi.
    SimpleMatrix rollingT = SimpleMatrix.identity(K);
    SimpleMatrix effectivePi = new SimpleMatrix(K,K);
    for(int i = 0; i < L-1; i++) {
      effectivePi = effectivePi.plus(rollingT.scale(1. / (L - 1)));
      rollingT = rollingT.mult(model.getT().transpose());
    }
    SimpleMatrix pi = model.getPi().transpose();
    effectivePi = pi.mult(effectivePi);
    assert(MatrixOps.equal( effectivePi.elementSum(), 1.0 ) );

    for(int h1 = 0; h1 < K; h1++) {
      for(int h2 = 0; h2 < K; h2++) {
        // pi (I + T + T^2...) * pi
        P[h1][h2] = effectivePi.get(h1) * T[h1][h2];
      }
    }

    return P;
  }

  public boolean piecewiseIdentifiabilityReport(HiddenMarkovModel model, int[][] X) {
    begin_track("piecewise-identifiability");
    int K = model.getStateCount();
    int D = model.getEmissionCount();

    // Piecewise likelihood.
    double[][] O = model.params.O;
    double[][] P = getPiecewiseParameters(model);

    // Compute hessian of likelihood
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

    assert(H.length == H[0].length);
    SimpleMatrix sigma = MatrixOps.svd(new SimpleMatrix(H)).getValue1().extractDiag().transpose();
    log(sigma);
    int rank = MatrixOps.rank(new SimpleMatrix(H));
    logs("Problem has rank %d vs %d.", rank, H.length);
    Execution.putOutput("piecewise-sigmak", sigma.get(H.length - 1));
    Execution.putOutput("piecewise-K", sigma.get(0,0) / sigma.get(H.length-1));

    end_track("piecewise-identifiability");

    return rank == H.length;
  }

  public double[][] recoverT(HiddenMarkovModel model, int[][] X, double[][] O) {
    LogInfo.begin_track("recover-T");
    int K = model.getStateCount();
    int D = model.getEmissionCount();

    // Initialization
    double[][] P;
    if(initExact) {
      P = getPiecewiseParameters(model);
    } else {
      P = RandomFactory.rand_(initRandom, K,K);
      MatrixOps.abs(P);
      MatrixOps.scale(P, 1./MatrixOps.sum(P));
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

    end_track("recover-T");

    SimpleMatrix T_ = new SimpleMatrix(T).transpose();
    T_ = MatrixOps.alignMatrix( T_, model.getT(), true );

    log(outputList(
            "T-error", MatrixOps.diff(T_, model.getT()),
            "\nT^", T_,
            "\nT*", model.getT()
    ));

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

    {
      SimpleMatrix piHat = MatrixFactory.fromVector(pi);
      piHat = MatrixOps.alignMatrix( piHat, model.getPi().transpose(), false );
      log(outputList(
              "pi-error", MatrixOps.diff(piHat, model.getPi().transpose()),
              "\npi^", piHat,
              "\npi*", model.getPi().transpose()
      ));
    }

    end_track("recover-pi");
    return pi;
  }

  public double[][] recoverO(HiddenMarkovModel model, int[][] X) {
    // -- Compose moments
    int K = model.getStateCount();
    int D = model.getEmissionCount();

    // Now factorize!
    begin_track("recover-O");
    double[][] O;
    SimpleMatrix O_;
    if(initExactO) {
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

    return O;
  }

  public HiddenMarkovModel spectralConvexRecovery(HiddenMarkovModel model, int[][] X) {
    // -- Compose moments
    int K = model.getStateCount();
    int D = model.getEmissionCount();

    double[][] O = recoverO(model, X);

    // Compute pi with maximum likelihood.
    double[] pi = recoverPi(model, X, O);

    double[][] T = recoverT(model, X, O);

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
    Execution.putOutput("true-pi", model.getPi());
    Execution.putOutput("true-T", model.getT());
    Execution.putOutput("true-O", model.getO());

    {
      SimpleMatrix O = model.getO();
      SimpleMatrix sigma = MatrixOps.svd(O).getValue1();
      Execution.putOutput("O-sigmak", sigma.get(K - 1, K - 1));
      Execution.putOutput("O-K", sigma.get(0, 0) / sigma.get(K - 1, K - 1));
    }
    fullIdentifiabilityReport(model, X);
    piecewiseIdentifiabilityReport(model, X);

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
      } break;
      case SpectralConvex: {
        model_ = spectralConvexRecovery(model, X);
      } break;
      default:
        throw new RuntimeException("Not implemented");
    }
    {
      // Align with true parameters.
      SimpleMatrix O = model_.getO();
      SimpleMatrix O_ = MatrixOps.alignMatrix(O, model.getO(), true);
      SimpleMatrix T = model_.getT();
      SimpleMatrix T_ = MatrixOps.alignMatrix(T, model.getT(), true);
      SimpleMatrix pi = model_.getPi();
      SimpleMatrix pi_ = MatrixOps.alignMatrix(pi, model.getPi(), false);
      log(outputList(
              "pi-error", MatrixOps.diff(pi_, model.getPi()),
              "T-error", MatrixOps.diff(T_, model.getT()),
              "O-error", MatrixOps.diff(O_, model.getO()),
              "params-error", model_.toParams().computeDiff(params, null)
      ));

      Execution.putOutput("pi-error", MatrixOps.diff(pi_, model.getPi()));
      Execution.putOutput("T-error", MatrixOps.diff(T_, model.getT()));
      Execution.putOutput("O-error", MatrixOps.diff(O_, model.getO()));

      Execution.putOutput("initial-likelihood", model_.likelihood(X));
      Execution.putOutput("initial-paramsError", model_.toParams().computeDiff(params, null));
      Execution.putOutput("initial-pi", model_.getPi());
      Execution.putOutput("initial-T", model_.getT());
      Execution.putOutput("initial-O", model_.getO());
    }

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
