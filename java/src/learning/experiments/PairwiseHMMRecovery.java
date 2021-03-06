package learning.experiments;

import fig.basic.*;
import fig.exec.Execution;
import learning.common.Counter;
import learning.linalg.FullTensor;
import learning.linalg.MatrixFactory;
import learning.linalg.MatrixOps;
import learning.linalg.RandomFactory;
import learning.models.DirectedGridModel;
import learning.models.HiddenMarkovModel;
import learning.models.loglinear.Example;
import learning.spectral.TensorMethod;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleSVD;
import org.javatuples.Quartet;

import javax.xml.ws.soap.MTOM;
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

  @Option( gloss = "number of states" )
  public int K = 2;
  @Option( gloss = "number of symbols" )
  public int D = 2;

  @Option( gloss = "Generator for parameters" )
  public Random paramsRandom = new Random(1);
  @Option( gloss = "variance for parameters" )
  public double paramsNoise = 1.0;

  @Option( gloss = "Generator for data" )
  public Random genRandom = new Random(1);

  @Option(gloss="Generator for initial points")
  public Random initRandom = new Random(42);
  @Option(gloss="Variance")
  public double initRandomNoise = 1.0;

  @Option(gloss="Start at the exact solution?")
  public boolean initExact = false;
  @Option(gloss="Start at the exact solution?")
  public boolean initExactO = false;


  @Option(gloss="iterations to run EM")
  public int iters = 100;

  @Option(gloss="Run EM?")
  public boolean runEM = false;
  @Option(gloss="Precision to run EM for")
  public double emEps = 1e-4;

  @Option(gloss="Project measurements?")
  public boolean projectMeasurements = true;
  @Option(gloss="Scale smoothing?")
  public boolean scaleSmoothing = false;
  @Option(gloss="How much to smooth")
  public double smoothMeasurements = 1e-3;

  @Option(gloss="Type of optimization to use") public boolean useLBFGS = false;
  @OptionSet(name="lbfgs") public LBFGSMaximizer.Options lbfgs = new LBFGSMaximizer.Options();
  @OptionSet(name="backtrack") public BacktrackingLineSearch.Options backtrack = new BacktrackingLineSearch.Options();

  enum RunMode {
    EM,
    NaiveSpectral,
    Spectral,
    Piecewise
  }
  @Option
  public RunMode mode = RunMode.EM;

  Maximizer newMaximizer() {
    if (useLBFGS) return new LBFGSMaximizer(backtrack, lbfgs);
    return new GradientMaximizer(backtrack);
  }

  private class Analysis {
    public final HiddenMarkovModel model;
    public final HiddenMarkovModel.Parameters trueParams;
    public final Counter<Example> data;

    public Analysis(HiddenMarkovModel model, HiddenMarkovModel.Parameters trueParams, Counter<Example> data) {
      this.model = model;
      this.trueParams = trueParams;
      this.data = data;

      initialReport();
    }

    public void initialReport() {
      SimpleMatrix O = new SimpleMatrix(trueParams.getO());
      SimpleMatrix sigma = MatrixOps.svd(O).getValue1();
      log( outputList(
              "O-sigmak", sigma.get(K - 1, K - 1),
              "O-K", sigma.get(0, 0) / sigma.get(K - 1, K - 1)
      ));
    }

    public void reportO(double[][] O_) {
      double[][] O = trueParams.getO();

      // Compute the difference.
      O = MatrixOps.alignRows(O, O_);
      double diff = MatrixOps.diff(O, O_);

      log(outputList(
              "O*", Fmt.D(O, "\n"),
              "\nO^", Fmt.D(O_, "\n"),
              "\ndO", diff
      ));
    }

    public void reportT(double[][] T_) {
      double[][] T = trueParams.getT();

      // Compute the difference.
      T = MatrixOps.alignRows(T, T_);
      double diff = MatrixOps.diff(T, T_);

      log(outputList(
              "T*", Fmt.D(T, "\n"),
              "\nT^", Fmt.D(T_, "\n"),
              "\ndT", diff
      ));
    }

    public void reportPi(double[] pi_) {
      double[] pi = trueParams.getPi();

      // Compute the difference.
      pi = MatrixOps.alignRows(pi, pi_);
      double diff = MatrixOps.diff(pi, pi_);

      log(outputList(
              "Pi*", Fmt.D(pi),
              "\nPi^", Fmt.D(pi_),
              "\ndPi", diff
      ));
    }

    public void reportParams(HiddenMarkovModel.Parameters params) {
      log(outputList(
              "theta*", trueParams,
              "\ntheta^", params,
              "\ndtheta", params.computeDiff2(trueParams,null)
      ));

      log(outputList(
              "pi-lhood*", piLikelihood(model, trueParams, data),
              "pi-lhood^",  piLikelihood(model, params, data),
              "T-lhood*", piecewiseLikelihood(model, trueParams, data),
              "T-lhood^", piecewiseLikelihood(model, params, data),
              "lhood*", model.getLogLikelihood(trueParams, data),
              "lhood^", model.getLogLikelihood(params, data)
      ));

    }

    public void reportInitialParams(HiddenMarkovModel.Parameters params) {
      log(outputList(
              "theta*", trueParams,
              "\ntheta0", params,
              "\ndtheta0", params.computeDiff2(trueParams,null)
      ));


      log(outputList(
              "pi-lhood*", piLikelihood(model, trueParams, data),
              "pi-lhood0 ",  piLikelihood(model, params, data),
              "T-lhood*", piecewiseLikelihood(model, trueParams, data),
              "T-lhood0 ", piecewiseLikelihood(model, params, data),
              "lhood*", model.getLogLikelihood(trueParams, data),
              "\nlhood0", model.getLogLikelihood(params, data)
      ));
    }
  }
  private Analysis analysis;

  public FullTensor computeMomentsO(HiddenMarkovModel model, Counter<Example> data) {
    int D = model.getD();
    double[][][] moments = new double[D][D][D];
//    int[][][] counts = new int[D][D][D];
//    int it = 0;
    for(Example ex : data) {
      double weight = data.getFraction(ex);
      int[] x = ex.x;
      // Sliding window
      for(int i = 0; i < x.length - 2; i++ ) {
//        it++;
        int x1 = x[i+0];
        int x2 = x[i+2];
        int x3 = x[i+1]; // Because we want the middle entry to be M3.
        // Update the moments -> All the values from counts to now (it) were zero
//        int updates = it - counts[x1][x2][x3];
        moments[x1][x2][x3] += weight;
//                (1. - updates * moments[x1][x2][x3]) / it;
//        counts[x1][x2][x3] = it;
      }
    }
//    // Final counting of zeros
//    for(int x1 = 0; x1 < D; x1++) {
//      for(int x2 = 0; x2 < D; x2++) {
//        for(int x3 = 0; x3 < D; x3++) {
//          int updates = it - counts[x1][x2][x3];
//          moments[x1][x2][x3] += (-updates * moments[x1][x2][x3]) / it;
//          counts[x1][x2][x3] = it;
//        }
//      }
//    }
    // Some checksums
    double sum = MatrixOps.sum(moments);
    MatrixOps.scale(moments, 1./sum);

    return new FullTensor(moments);
  }

  public Pair<SimpleMatrix,SimpleMatrix> recoverOT(HiddenMarkovModel model, Counter<Example> data) {
    // -- Compose moments
    // Now factorize!
    begin_track("recover-O");

    FullTensor momentsO = computeMomentsO(model, data);
    TensorMethod tensorMethod = new TensorMethod();
    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, SimpleMatrix> params = tensorMethod.recoverParametersAsymmetric(K, momentsO);

    // Aligned by K
    SimpleMatrix O = params.getValue3();
    SimpleMatrix OT = params.getValue2();

    if(projectMeasurements)
      O = MatrixOps.projectOntoSimplex(O, smoothMeasurements);
    SimpleMatrix T = O.pseudoInverse().mult(OT);
    if(projectMeasurements)
      T = MatrixOps.projectOntoSimplex(T, smoothMeasurements);

    end_track("recover-O");

    return Pair.newPair(O,T);
  }

  public double[][] getPiecewiseParameters(HiddenMarkovModel model, HiddenMarkovModel.Parameters params) {
    int K = model.getK();
    SimpleMatrix T = new SimpleMatrix(params.getT());
    SimpleMatrix pi = MatrixFactory.fromVector(params.getPi());

    // Preprocess pi.
    SimpleMatrix rollingT = SimpleMatrix.identity(K);
    SimpleMatrix effectivePi = new SimpleMatrix(K,K);
    for(int i = 0; i < L-1; i++) {
      effectivePi = effectivePi.plus(rollingT.scale(1. / (L - 1)));
      rollingT = rollingT.mult(T);
    }
    effectivePi = pi.mult(effectivePi);
    assert(MatrixOps.equal( effectivePi.elementSum(), 1.0 ) );

    double[][] P = new double[K][K];
    for(int h1 = 0; h1 < K; h1++) {
      for(int h2 = 0; h2 < K; h2++) {
        // pi (I + T + T^2...) * pi
        P[h1][h2] = effectivePi.get(h1) * T.get(h1,h2);
      }
    }

    return P;
  }


//
//  public double[][] computeLikelihoodHessian(HiddenMarkovModelOld model, int[][] X) {
//    final double eps = 1e-6;
//    double[] weights = model.params.toVector();
//    double[] original = weights.clone();
//    double[][] H = new double[weights.length][weights.length];
//
//    double lhood = model.likelihood(X);
//    for(int i = 0; i < weights.length; i++) {
//      for(int j = 0; j < weights.length; j++) {
//        weights[i] += eps;
//        weights[j] += eps;
//        model.params.updateFromVector(weights);
//        double lhoodPlusPlus = model.likelihood(X);
//
//        weights[j] -= 2*eps;
//        model.params.updateFromVector(weights);
//        double lhoodPlusMinus = model.likelihood(X);
//
//        weights[i] -= 2*eps;
//        model.params.updateFromVector(weights);
//        double lhoodMinusMinus = model.likelihood(X);
//
//        weights[j] += 2*eps;
//        model.params.updateFromVector(weights);
//        double lhoodMinusPlus = model.likelihood(X);
//
//        weights[i] += eps;
//        weights[j] -= eps;
//
//        H[i][j] = ((lhoodPlusPlus - lhoodMinusPlus)/(2*eps) + (lhoodMinusMinus  - lhoodPlusMinus)/(2*eps))/(2*eps);
//      }
//    }
//    model.params.updateFromVector(original);
//
//    return H;
//  }
//
//  public boolean fullIdentifiabilityReport(HiddenMarkovModelOld model, int[][] X) {
//    int K = model.getStateCount();
//    int D = model.getEmissionCount();
//
//    double[][] H = computeLikelihoodHessian(model, X);
//
//    assert(H.length == H[0].length);
//    SimpleMatrix sigma = MatrixOps.svd(new SimpleMatrix(H)).getValue1().extractDiag().transpose();
//    log(sigma);
//    int rank = MatrixOps.rank(new SimpleMatrix(H));
//    logs("Problem has rank %d vs %d.", rank, H.length);
//    Execution.putOutput("full-sigmak", sigma.get(H.length-1));
//    Execution.putOutput("full-K", sigma.get(0,0) / sigma.get(H.length-1));
//    return rank == H.length;
//  }

  public SimpleMatrix getPiecewiseHessian(HiddenMarkovModel model, HiddenMarkovModel.Parameters params, double[][] P, Counter<Example> data)  {
    int K = model.getK();
    int D = model.getD();

    // Piecewise likelihood.
    double[][] O = params.getO();

    // Compute hessian of likelihood
    double[][] H = new double[K*K][K*K];

    int count = 0;
    for(Example ex : data) {
      double weight = data.getFraction(ex);
      int[] x = ex.x;

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

    return new SimpleMatrix(H);
  }

  public boolean piecewiseIdentifiabilityReport(HiddenMarkovModel model, HiddenMarkovModel.Parameters params, Counter<Example> data) {
    begin_track("piecewise-identifiability");
    int K = model.getK();
    int D = model.getD();

    double Sigmak = Double.POSITIVE_INFINITY;

    boolean identifiable = true;
    {
      double[][] P = getPiecewiseParameters(model, analysis.trueParams);
      SimpleMatrix H = getPiecewiseHessian(model, params, P, data);
      int p = H.numRows();

      SimpleSVD svd = H.svd();
      int rank = MatrixOps.rank(H);
      double sigmak = svd.getSingleValue(p-1);
      double condition = svd.getSingleValue(0) / sigmak;
      logs("Problem has rank %d vs %d (%f).", rank, p, condition);
      log(svd.getW());

      if(sigmak < Sigmak) Sigmak = sigmak;
      identifiable = identifiable && K < 1e3;
    }
//    for(int guess = 0; guess < 5; guess++) {
//      double[][] P = RandomFactory.rand_(initRandom, K, K);
//      MatrixOps.abs(P);
//      MatrixOps.scale(P, 1./MatrixOps.sum(P));
//      SimpleMatrix H = getPiecewiseHessian(model, P, X);
//      int p = H.numRows();
//
//      SimpleSVD svd = H.svd();
//      int rank = svd.rank();
//      double sigmak = svd.getSingleValue(p-1);
//      double condition = svd.getSingleValue(0) / sigmak;
//      logs("Problem has rank %d vs %d (%f).", svd.rank(), p, condition);
//      log(svd.getW());
//
//      if(sigmak < Sigmak) Sigmak = sigmak;
//      identifiable = identifiable && K < 1e3;
//    }
    end_track("piecewise-identifiability");

    Execution.putOutput("piecewise-sigmak", Sigmak);

    return identifiable;
  }
//
//  public SimpleMatrix piecewiseLikelihoodGradient(HiddenMarkovModelOld model, double[][] P, double[][] O, int[][] X) {
//    int K = model.getStateCount();
//    int count = 0;
//    double[][] gradient = new double[P.length][P.length];
//    for(int[] x : X) {
//      for(int i = 0; i < x.length-1;i++) {
//        count++;
//
//        int x1 = x[i+0];
//        int x2 = x[i+1];
//
//        double z = 0.;
//        for(int h1 = 0; h1 < K; h1++)
//          for(int h2 = 0; h2 < K; h2++)
//            z += P[h1][h2] * O[h1][x1] * O[h2][x2];
//
//        // Update
//        for(int h1 = 0; h1 < K; h1++)
//          for(int h2 = 0; h2 < K; h2++)
//            gradient[h1][h2] += ((O[h1][x1] * O[h2][x2]/z) - gradient[h1][h2])/(count);
//      }
//    }
//
//    return new SimpleMatrix(gradient);
//  }

  public double piLikelihood(HiddenMarkovModel model, HiddenMarkovModel.Parameters params, Counter<Example> data) {
    int K = model.getK();
    int D = model.getD();

    double[][] O = params.getO();
    double[] pi = params.getPi();

    double lhood = 0;
    for(Example ex : data) {
      double weight = data.getFraction(ex);
      int[] x = ex.x;
      int x1 = x[0];

      double z = 0.;
      for(int h1 = 0; h1 < K; h1++)
        z += pi[h1] * O[h1][x1];

      // Update
      assert z > 0;
      lhood += weight * Math.log(z);
    }

    return lhood;
  }


  public double piecewiseLikelihood(HiddenMarkovModel model, HiddenMarkovModel.Parameters params, Counter<Example> data) {
    int K = model.getK();
    int D = model.getD();


    double[][] O = params.getO();
    double[][] P = getPiecewiseParameters(model, params);

    double lhood = 0;
    for(Example ex : data) {
      double weight = data.getFraction(ex);
      int[] x = ex.x;
      for(int i = 0; i < x.length-1;i++) {
        int x1 = x[i+0];
        int x2 = x[i+1];

        double z = 0.;
        for(int h1 = 0; h1 < K; h1++)
          for(int h2 = 0; h2 < K; h2++)
            z += P[h1][h2] * O[h1][x1] * O[h2][x2];

        // Update
        assert z > 0;
        lhood += weight * Math.log(z);
      }
    }

    return lhood;
  }

  public double[][] recoverP(HiddenMarkovModel model, double[][] O, Counter<Example> data) {
    LogInfo.begin_track("recover-P");
    int K = model.getK();
    int D = model.getD();

    // Initialization
    double[][] P;
    P = RandomFactory.rand_(initRandom, K,K);
    MatrixOps.abs(P);
    MatrixOps.scale(P, 1./MatrixOps.sum(P));
    assert( MatrixOps.equal(MatrixOps.sum(P), 1.0) );

    // Temporary buffer
    boolean done = false;

    double lhood_old = Double.NEGATIVE_INFINITY;
    for(int iter = 0; iter < 1000 && !done; iter++){
      // Compute marginals
      double lhood = 0.;
      double[][] P_ = new double[K][K];
      for(Example ex : data) {
        double weight = data.getFraction(ex);
        int[] x = ex.x;
        for(int i = 0; i < x.length-1;i++) {
          int x1 = x[i+0];
          int x2 = x[i+1];

          double z = 0.;
          for(int h1 = 0; h1 < K; h1++)
            for(int h2 = 0; h2 < K; h2++)
              z += P[h1][h2] * O[h1][x1] * O[h2][x2];

          // Update
          assert z > 0;
          lhood += weight * Math.log(z);
          for(int h1 = 0; h1 < K; h1++)
            for(int h2 = 0; h2 < K; h2++)
              P_[h1][h2] += (weight * P[h1][h2] * O[h1][x1] * O[h2][x2]) / z;
        }
      }
      double z = MatrixOps.sum(P_);
      assert (MatrixOps.equal(z, L - 1)); // but maybe not exactly because of numerical error
      // Normalize
      lhood /= (L-1);
      for(int h1 = 0; h1 < K; h1++) {
        for(int h2 = 0; h2 < K; h2++) {
          P_[h1][h2] /= z;
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

      LogInfo.dbg(outputList(
              "iter", iter,
              "lhood", lhood,
              "diff", diff
      ));

      done = diff < emEps;
    }
    LogInfo.log(outputList(
            "lhood", lhood_old
    ));

    end_track("recover-P");

    return P;
  }

  public double[][] recoverT(HiddenMarkovModel model, double[][] O, Counter<Example> data) {
    LogInfo.begin_track("recover-T");
    int K = model.getK();

    // Initialization
    double[][] P = recoverP(model, O, data);

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

    return T;
  }

  public double[] recoverPi(HiddenMarkovModel model, double[][] O, Counter<Example> data) {
    LogInfo.begin_track("recover-pi");
    int K = model.getK();
    int D = model.getD();

    // Initialization
    HiddenMarkovModel.Parameters params = model.newParams();
    params.initRandom(initRandom, 1.0);
    double[] pi = params.getPi();

    // Temporary buffer
    boolean done = false;

    double lhood_old = Double.NEGATIVE_INFINITY;

    for(int iter = 0; iter < 1000 && !done; iter++){
      // Compute marginals
      double lhood = 0.;
      // Marginals
      double[] pi_ = new double[K];
      for(Example ex : data) {
        double weight = data.getFraction(ex);
        int[] x = ex.x;
        int x1 = x[0];

        double z = 0.;
        for(int h1 = 0; h1 < K; h1++)
            z += pi[h1] * O[h1][x1];

        // Update
        assert z > 0;
        lhood += weight * Math.log(z);
        for(int h1 = 0; h1 < K; h1++)
          pi_[h1] += weight * (pi[h1] * O[h1][x1] / z);
      }
      double z = MatrixOps.sum(pi);
      assert (MatrixOps.equal(z, 1)); // but maybe not exactly because of numerical error
      // Update
      double diff = 0.;
      for(int h1 = 0; h1 < K; h1++) {
        diff += Math.abs(pi[h1] - pi_[h1]);
        pi[h1] = pi_[h1];
      }
      assert( MatrixOps.equal(MatrixOps.sum(pi), 1.0 ));

      assert(lhood > lhood_old);
      lhood_old = lhood;

      LogInfo.dbg(outputList(
              "iter", iter,
              "lhood", lhood,
              "diff", diff
      ));

      done = diff < emEps;
    }
    LogInfo.log(outputList(
            "lhood", lhood_old
    ));

    end_track("recover-pi");

    return pi;
  }

  public double[] recoverPiSpectral(HiddenMarkovModel model, double[][] O, Counter<Example> data) {
    LogInfo.begin_track("recover-pi");
    // Collect second order moments

    double[] M1 = new double[D];
    for(Example ex: data) {
      M1[ex.x[0]] += data.getFraction(ex);
    }
    assert MatrixOps.equal(MatrixOps.sum(M1), 1.0);


    // Now invert O to get pi.
    double[] pi;
    {
      SimpleMatrix O_ = new SimpleMatrix(O);
      SimpleMatrix M1_ = MatrixFactory.fromVector(M1);
      SimpleMatrix pi_ = M1_.mult(O_.pseudoInverse());
      // Project on to simplex
      if (projectMeasurements)
        pi_ = MatrixOps.projectOntoSimplex( pi_.transpose(), smoothMeasurements );

      pi = MatrixFactory.toVector(pi_);
    }


    // Normalize pi
    MatrixOps.scale(pi, 1./MatrixOps.sum(pi));

    LogInfo.end_track("recover-pi");

    return pi;
  }

  public HiddenMarkovModel.Parameters piecewiseRecovery(HiddenMarkovModel model, Counter<Example> data) {
    Pair<SimpleMatrix,SimpleMatrix> OT = recoverOT(model, data);
    double[][] O = MatrixFactory.toArray(OT.getFirst().transpose());
    analysis.reportO(O);

    // Compute T with convex EM
    double[][] T = recoverT(model, O, data);
    analysis.reportT(T);

    // Compute pi with maximum likelihood.
    double[] pi = recoverPi(model, O, data);
    analysis.reportPi(pi);

    return model.newParams().with(pi, T, O);
  }

  public HiddenMarkovModel.Parameters spectralRecovery(HiddenMarkovModel model, Counter<Example> data) {
    // -- Compose moments
    Pair<SimpleMatrix,SimpleMatrix> OT = recoverOT(model, data);
    double[][] O = MatrixFactory.toArray(OT.getFirst().transpose());
    double[][] T = MatrixFactory.toArray(OT.getSecond().transpose());

    analysis.reportO(O);
    analysis.reportT(T);

    // Compute pi with maximum likelihood.
    double[] pi = recoverPiSpectral(model, O, data);
    analysis.reportPi(pi);

    return model.newParams().with(pi, T, O);
  }

//  public void sanityCheck(HiddenMarkovModelOld model, int[][] X) {
//    int K = model.getStateCount();
//    double[][] O = model.params.O;
//    double[][] P1 = recoverP(model, X, O);
//    double[][] P2 = recoverP(model, X, O);
//
//    log(Fmt.D(P1));
//    log(Fmt.D(P2));
//
//    double lhood1 = piecewiseLikelihood(model, P1, O, X);
//    double lhood2 = piecewiseLikelihood(model, P2, O, X);
//    log("Likelihood1: " + lhood1);
//    log("Likelihood2: " + lhood2);
//
//    for(int i = 0; i < 10; i++) {
//      double[][] P3 = new double[P1.length][P1.length];
//      double eta = i/9.;
//      for(int h1 = 0; h1 < K ; h1++) {
//        for(int h2 = 0; h2 < K ; h2++) {
//          P3[h1][h2] = (1-eta) * P1[h1][h2]  + eta * P2[h1][h2];
//        }
//      }
//      double lhood = piecewiseLikelihood(model,P3, O, X);
//      logs("Likelihood (%.2f): %f", eta, lhood);
//    }
//
//    SimpleMatrix G1 = piecewiseLikelihoodGradient(model, P1, O, X);
//    SimpleMatrix G2 = piecewiseLikelihoodGradient(model, P2, O, X);
//
//    SimpleMatrix H1 = getPiecewiseHessian(model, P1, X);
//    SimpleMatrix H2 = getPiecewiseHessian(model, P2, X);
//
//    log( G1 );
//    log( G2 );
//
//    log( H1 );
//    log( H2 );
//
//    double lhood1_ = lhood2;
//    double lhood2_ = lhood1;
//
//    for(int i = 0; i < K * K; i++) {
//      int h1 = i / K;
//      int h2 = i % K;
//
//      //
//      lhood2_ += (P2[h1][h2] - P1[h1][h2]) * G1.get(h1,h2);
//      lhood1_ += (P1[h1][h2] - P2[h1][h2]) * G2.get(h1,h2);
//
//      for(int j = 0; j < K * K; j++) {
//        int h1_ = j / K;
//        int h2_ = j % K;
//
//        // Add O_z
//        lhood2_ += (P2[h1][h2] - P1[h1][h2]) * H1.get(i,j) * (P2[h1_][h2_] - P1[h1_][h2_]);
//        lhood1_ += (P1[h1][h2] - P2[h1][h2]) * H2.get(i,j) * (P1[h1_][h2_] - P2[h1_][h2_]);
//      }
//    }
//    log("Estimated Likelihood1: " + lhood1_);
//    log("Estimated Likelihood2: " + lhood2_);
//  }

  @Override
  public void run() {
    // Options pre-processing
    if(projectMeasurements && scaleSmoothing) {
      if(N >= 1e7)
        projectMeasurements = false;
      else
        smoothMeasurements /= Math.sqrt(N);
    }

    HiddenMarkovModel model = new HiddenMarkovModel(K, D, L);
    // Initialize model
    begin_track("Generating model");
    HiddenMarkovModel.Parameters trueParams = model.newParams();
    
    double sigmak = 0.;
    while(sigmak < 0.05) {
        trueParams.initRandom(paramsRandom, paramsNoise);
        SimpleMatrix sigma = MatrixOps.svd(new SimpleMatrix(trueParams.getO())).getValue1();
        sigmak = sigma.get(K-1,K-1);
    }

    // Get data
    Counter<Example> data;
    if(N >= 1e7) // Seriously too much data
      data =  model.getDistribution(trueParams);
    else
      data =  model.drawSamples(trueParams, genRandom, (int) N);
    analysis = new Analysis(model, trueParams, data);

    end_track("Generating model");

    // TODO: Make this less sucky
//    fullIdentifiabilityReport(model, X);
    piecewiseIdentifiabilityReport(model, trueParams, data);

    begin_track("Get initial estimate");

    // Process via EM or Spectral
    HiddenMarkovModel.Parameters params;
    switch(mode) {
      case EM: {
        params = model.newParams();
        if(initExact)
          params.copyOver(trueParams);
        else
          params.initRandom(initRandom, initRandomNoise);
        runEM = true; // Regardless of what you said before.
      } break;
      case Spectral: {
        params = spectralRecovery(model, data);
      } break;
      case Piecewise: {
        params = piecewiseRecovery(model, data);
      } break;
      default:
        throw new RuntimeException("Not implemented");
    }
    analysis.reportInitialParams(params);
    end_track("Get initial estimate");

    if(runEM) {
      begin_track("Run em");

      HiddenMarkovModel.Parameters marginals = model.newParams();
      double lhood_ = Double.NEGATIVE_INFINITY;
      for(int i = 0; i < 1000; i++) {
        double lhood = model.getLogLikelihood(params,data);

        // Simple EM
        marginals.clear();
        model.updateMarginals(params, data, 1.0, marginals);
        double diff = params.computeDiff2(marginals, null);

        dbg(outputList(
                "iter", i,
                "likelihood", lhood,
                "diff", diff
        ));
        //assert( lhood - lhood_ > -1e-3); // Numerical error.
        if( diff < emEps ) break;

        lhood_ = lhood;
        params.copyOver(marginals);
      }

      log(outputList(
              "likelihood", lhood_
      ));

      analysis.reportO(params.getO());
      analysis.reportT(params.getT());
      analysis.reportPi(params.getPi());

      analysis.reportParams(params);

      end_track("Run em");
    }
  }

  public static void main(String[] args) {
    Execution.run(args, new PairwiseHMMRecovery());
  }
}
