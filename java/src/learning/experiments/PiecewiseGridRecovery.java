package learning.experiments;

import fig.basic.Option;
import fig.exec.Execution;
import learning.common.Counter;
import learning.linalg.MatrixOps;
import learning.models.BasicParams;
import learning.models.Params;
import learning.models.loglinear.Example;
import learning.models.DirectedGridModel;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleSVD;

import java.util.Random;

import static fig.basic.LogInfo.*;
import static learning.common.Utils.outputList;

/**
 * Exactly what it says.
 */
public class PiecewiseGridRecovery implements Runnable {
  @Option(gloss="Data used")
  public double N = 1e2;
  @Option(gloss="Sequence length")
  public int L = 4;

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

  /**
   * P(h1,h2,h3,h4) = pi(h1) T(h2|h1) T(h3|h1) T(h4|h2,h3)
   * @param model
   * @param params
   * @return
   */
  public double[][][][] getPiecewiseParameters(DirectedGridModel model, DirectedGridModel.Parameters params) {
    double[][][][] P = new double[K][K][K][K];

    for(int h1 = 0; h1 < K; h1++)
      for(int h2 = 0; h2 < K; h2++)
        for(int h3 = 0; h3 < K; h3++)
          for(int h4 = 0; h4 < K; h4++)
            P[h1][h2][h3][h4] = params.weights[model.pi(h1)] * params.weights[model.t(h1, h2)]
                    * params.weights[model.t(h1, h3)] * params.weights[model.tC(h2, h3, h4)];

    assert MatrixOps.equal(MatrixOps.sum(P), 1.0);

    return P;
  }

  public double[][] getEdgeParameters(DirectedGridModel model, DirectedGridModel.Parameters params) {
    double[][] P = new double[K][K];

    for(int h1 = 0; h1 < K; h1++)
      for(int h2 = 0; h2 < K; h2++)
          // Sum over the h1
          P[h1][h2] += params.weights[model.pi(h1)] * params.weights[model.t(h1, h2)];

    assert MatrixOps.equal(MatrixOps.sum(P), 1.0);

    return P;
  }

  public double[][][] getVStructureParameters(DirectedGridModel model, DirectedGridModel.Parameters params) {
    double[][][] P = new double[K][K][K];

    for(int h1 = 0; h1 < K; h1++)
      for(int h2 = 0; h2 < K; h2++)
        for(int h3 = 0; h3 < K; h3++)
          for(int h4 = 0; h4 < K; h4++)
            // Sum over the h1
            P[h2][h3][h4] += params.weights[model.pi(h1)] * params.weights[model.t(h1, h2)]
                    * params.weights[model.t(h1, h3)] * params.weights[model.tC(h2, h3, h4)];

    assert MatrixOps.equal(MatrixOps.sum(P), 1.0);

    return P;
  }

  /**
   * grad2 = o1 x o2 x o3 x o4
   * @param model
   * @param P
   * @param data
   * @return
   */
  public SimpleMatrix getPiecewiseHessian(DirectedGridModel model, DirectedGridModel.Parameters trueParams,
                                          double[][][][] P, Counter<Example> data) {
    double[][] H = new double[K*K*K*K][K*K*K*K];

    double[][] O = trueParams.getO();

    for(Example ex: data) {
      double weight = data.getFraction(ex);

      int x1 =  ex.x[model.oIdx(0,0,0)];
      int x1_ = ex.x[model.oIdx(0,0,1)];
      int x2 =  ex.x[model.oIdx(0,1,0)];
      int x2_ = ex.x[model.oIdx(0,1,1)];
      int x3 =  ex.x[model.oIdx(1,0,0)];
      int x3_ = ex.x[model.oIdx(1,0,1)];
      int x4 =  ex.x[model.oIdx(1,1,0)];
      int x4_ = ex.x[model.oIdx(1,1,1)];

      double z = 0.;
      for(int i = 0; i < K*K*K*K; i++) {
        int h1 = i % K;
        int h2 = (i / K) % K;
        int h3 = (i / (K*K)) % K;
        int h4 = (i / (K*K*K)) % K;

        z += P[h1][h2][h3][h4]
                *  O[h1][x1] * O[h1][x1_]
                *  O[h2][x2] * O[h2][x2_]
                *  O[h3][x3] * O[h3][x3_]
                *  O[h4][x4] * O[h4][x4_];
      }

      for(int i = 0; i < K*K*K*K; i++) {
        int h1 = i % K;
        int h2 = (i / K) % K;
        int h3 = (i / (K*K)) % K;
        int h4 = (i / (K*K*K)) % K;

        double v1 = O[h1][x1] * O[h1][x1_]
                    *  O[h2][x2] * O[h2][x2_]
                    *  O[h3][x3] * O[h3][x3_]
                    *  O[h4][x4] * O[h4][x4_]
                    / z;

        for(int j = 0; j < K*K*K*K; j++) {
          int h1_ =  j % K;
          int h2_ = (j / K) % K;
          int h3_ = (j / (K*K)) % K;
          int h4_ = (j / (K*K*K)) % K;

          double v2 = O[h1_][x1] * O[h1_][x1_]
                  *   O[h2_][x2] * O[h2_][x2_]
                  *   O[h3_][x3] * O[h3_][x3_]
                  *   O[h4_][x4] * O[h4_][x4_]
                  / z;

          H[i][j] += weight * v1 * v2;
        }
      }
    }
    // Compute outer product

    return new SimpleMatrix(H);
  }

  public SimpleMatrix getVStructureHessian(DirectedGridModel model, DirectedGridModel.Parameters trueParams,
                                          double[][][] P, Counter<Example> data) {
    double[][] H = new double[K*K*K][K*K*K];

    double[][] O = trueParams.getO();

    for(Example ex: data) {
      double weight = data.getFraction(ex);

      int x2 =  ex.x[model.oIdx(0,1,0)];
      int x2_ = ex.x[model.oIdx(0,1,1)];
      int x3 =  ex.x[model.oIdx(1,0,0)];
      int x3_ = ex.x[model.oIdx(1,0,1)];
      int x4 =  ex.x[model.oIdx(1,1,0)];
      int x4_ = ex.x[model.oIdx(1,1,1)];

      double z = 0.;
      for(int i = 0; i < K*K*K; i++) {
        int h2 = i % K;
        int h3 = (i / K) % K;
        int h4 = (i / (K*K)) % K;

        z += P[h2][h3][h4]
                *  O[h2][x2] * O[h2][x2_]
                *  O[h3][x3] * O[h3][x3_]
                *  O[h4][x4] * O[h4][x4_];
      }

      for(int i = 0; i < K*K*K; i++) {
        int h2 = (i ) % K;
        int h3 = (i / (K)) % K;
        int h4 = (i / (K*K)) % K;

        double v1 =  O[h2][x2] * O[h2][x2_]
                *  O[h3][x3] * O[h3][x3_]
                *  O[h4][x4] * O[h4][x4_]
                / z;

        for(int j = 0; j < K*K*K; j++) {
          int h2_ = (j) % K;
          int h3_ = (j / (K)) % K;
          int h4_ = (j / (K*K)) % K;

          double v2 = O[h2_][x2] * O[h2_][x2_]
                  *   O[h3_][x3] * O[h3_][x3_]
                  *   O[h4_][x4] * O[h4_][x4_]
                  / z;

          H[i][j] += weight * v1 * v2;
        }
      }
    }
    // Compute outer product

    return new SimpleMatrix(H);
  }

  public SimpleMatrix getEdgeHessian(DirectedGridModel model, DirectedGridModel.Parameters trueParams,
                                          double[][] P, Counter<Example> data) {
    double[][] H = new double[K*K][K*K];

    double[][] O = trueParams.getO();

    for(Example ex: data) {
      double weight = data.getFraction(ex);

      int x1 =  ex.x[model.oIdx(0,0,0)];
      int x1_ = ex.x[model.oIdx(0,0,1)];
      int x2 =  ex.x[model.oIdx(0,1,0)];
      int x2_ = ex.x[model.oIdx(0,1,1)];

      double z = 0.;
      for(int i = 0; i < K*K; i++) {
        int h1 = i % K;
        int h2 = (i / K) % K;

        z += P[h1][h2]
                *  O[h1][x1] * O[h1][x1_]
                *  O[h2][x2] * O[h2][x2_];

      }
      for(int i = 0; i < K*K; i++) {
        int h1 = i % K;
        int h2 = (i / K) % K;

        double v1 = O[h1][x1] * O[h1][x1_]
                *  O[h2][x2] * O[h2][x2_]
                / z;

        for(int j = 0; j < K*K; j++) {
          int h1_ =  j % K;
          int h2_ = (j / K) % K;

          double v2 = O[h1_][x1] * O[h1_][x1_]
                  *   O[h2_][x2] * O[h2_][x2_]
                  / z;

          H[i][j] += weight * v1 * v2;
        }
      }
    }
    // Compute outer product

    return new SimpleMatrix(H);
  }

  public boolean piecewiseIdentifiabilityReport(DirectedGridModel model, DirectedGridModel.Parameters params, Counter<Example> data) {
    begin_track("piecewise-identifiability");
    int K = model.getK();
    int D = model.getD();

    // Piecewise likelihood.
    boolean identifiable;
    {
      double[][][] P = getVStructureParameters(model, params);
      SimpleMatrix H = getVStructureHessian(model, params, P, data);
      int p = H.numRows();

      SimpleSVD svd = H.svd();
      int rank = MatrixOps.rank(H, 1e-3);
      double sigmak = svd.getSingleValue(p-1);
      double condition = svd.getSingleValue(0) / sigmak;
      logs("Problem has rank %d vs %d (%f).", rank, p, condition);
      log(svd.getW().extractDiag().transpose());

      log(outputList(
         "piecewise-V-sigmak", sigmak
      ));
      identifiable = K < 1e3;
    }
    {
      double[][] P = getEdgeParameters(model, params);
      SimpleMatrix H = getEdgeHessian(model, params, P, data);
      int p = H.numRows();

      SimpleSVD svd = H.svd();
      int rank = MatrixOps.rank(H, 1e-3);
      double sigmak = svd.getSingleValue(p-1);
      double condition = svd.getSingleValue(0) / sigmak;
      logs("Problem has rank %d vs %d (%f).", rank, p, condition);
      log(svd.getW().extractDiag().transpose());

      log(outputList(
              "piecewise-e-sigmak", sigmak
      ));
      identifiable = K < 1e3;
    }

    end_track("piecewise-identifiability");
    return identifiable;
  }



  @Override
  public void run() {
    DirectedGridModel model = new DirectedGridModel(K, D, L);
    // Initialize model
    begin_track("Generating model");
    DirectedGridModel.Parameters trueParams = model.newParams();
    trueParams.initRandom(paramsRandom, paramsNoise);

    // Get data
    Counter<Example> data;
    if(N >= 1e9) // Seriously too much data
      data =  model.getDistribution(trueParams);
    else
      data =  model.drawSamples(trueParams, genRandom, (int) N);
    end_track("Generating model");

    // Identifiability test
    piecewiseIdentifiabilityReport(model, trueParams, data);

  }

  public static void main(String[] args) {
    Execution.run(args, new PiecewiseGridRecovery());
  }

}
