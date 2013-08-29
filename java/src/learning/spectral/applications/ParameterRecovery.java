package learning.spectral.applications;

import fig.basic.LogInfo;
import learning.data.ComputableMoments;
import learning.data.HasExactMoments;
import learning.data.HasSampleMoments;
import learning.linalg.FullTensor;
import learning.linalg.MatrixFactory;
import learning.linalg.MatrixOps;
import learning.models.HiddenMarkovModel;
import learning.models.MixtureOfGaussians;
import learning.spectral.TensorMethod;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

/**
 * Canoncial class for parameter recovery using the "spectral" tensor factorization method.
 */
public class ParameterRecovery {
  static TensorMethod tensorMethod = new TensorMethod();

  static class HMMAnalysis {
    public HiddenMarkovModel trueModel;
    public HMMAnalysis(HiddenMarkovModel model) {
      trueModel = model;
    }

    public void compareMeasurements(Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> measurements) {
      Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, SimpleMatrix> trueMeasurements =  trueModel.computeMixtureViews();

      SimpleMatrix w = trueMeasurements.getValue0().transpose();
      SimpleMatrix w_ = measurements.getValue0();
      SimpleMatrix M1 = trueMeasurements.getValue1();
      SimpleMatrix M1_ = measurements.getValue1();
      SimpleMatrix M2 = trueMeasurements.getValue2();
      SimpleMatrix M2_ = measurements.getValue2();
      SimpleMatrix M3 = trueMeasurements.getValue3();
      SimpleMatrix M3_ = measurements.getValue3();

      int[] perm = MatrixOps.alignColumns(M1_, M1);
      w_ = MatrixOps.permuteColumns(w_, perm);
      M1_ = MatrixOps.permuteColumns(M1_, perm);
      M2_ = MatrixOps.permuteColumns(M2_, perm);
      M3_ = MatrixOps.permuteColumns(M3_, perm);

      LogInfo.logs("w error: " + MatrixOps.diff(w, w_));
      LogInfo.logs("M1 error: " + MatrixOps.diff(M1, M1_));
      LogInfo.logs("M2 error: " + MatrixOps.diff(M2, M2_));
      LogInfo.logs("M3 error: " + MatrixOps.diff(M3, M3_));
    }
  }
  public static HMMAnalysis hmmAnalysis;


  static HiddenMarkovModel recoverHMM(int K, Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> measurements, double smoothMeasurements) {
    LogInfo.begin_track("cast-conditional-means-to-hmm");

    if( hmmAnalysis != null ) {
      hmmAnalysis.compareMeasurements(measurements);
    }
    // Populate params
    SimpleMatrix Tpi = measurements.getValue0();
    // Ignoring M1 which is arbitrarily complicated.
    SimpleMatrix O = measurements.getValue2();
    SimpleMatrix OT = measurements.getValue3();
//    SimpleMatrix OT = measurements.getValue2();
//    SimpleMatrix O = measurements.getValue3();
    // construct O T and pi
    SimpleMatrix T = O.pseudoInverse().mult( OT );

    SimpleMatrix pi = T.pseudoInverse().mult(Tpi.transpose()).transpose(); // pi is pretty terrible...

    // project and smooth
    // projectOntoSimplex normalizes columns!
    pi = MatrixOps.projectOntoSimplex( pi.transpose(), smoothMeasurements );
    T = MatrixOps.projectOntoSimplex( T, smoothMeasurements ).transpose();
    O = MatrixOps.projectOntoSimplex( O, smoothMeasurements ).transpose();

    HiddenMarkovModel model = new HiddenMarkovModel( new HiddenMarkovModel.Params(
            MatrixFactory.toVector(pi),
            MatrixFactory.toArray(T),
            MatrixFactory.toArray(O)) );
    LogInfo.end_track("cast-conditional-means-to-hmm");

    return model;
  }

  public static HiddenMarkovModel recoverHMM(int K, ComputableMoments moments, double smoothMeasurements) {
    return recoverHMM(K, tensorMethod.randomizedRecoverParameters(K, moments), smoothMeasurements);
  }

  public static HiddenMarkovModel recoverHMM(int K, HasExactMoments moments, double smoothMeasurements) {
    return recoverHMM(K, tensorMethod.recoverParameters(K, moments.computeExactMoments()), smoothMeasurements);
  }

  public static HiddenMarkovModel recoverHMM(int K, int N, HasSampleMoments moments, double smoothMeasurements) {
    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> moments_ = moments.computeSampleMoments(N);
    if( moments instanceof HasExactMoments) {
      Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> momentsExact = ((HasExactMoments) moments).computeExactMoments();
      LogInfo.logs( "P13 error: " + MatrixOps.diff(moments_.getValue0(), momentsExact.getValue0() ) );
      LogInfo.logs( "P12 error: " + MatrixOps.diff(moments_.getValue1(), momentsExact.getValue1() ) );
      LogInfo.logs( "P32 error: " + MatrixOps.diff(moments_.getValue2(), momentsExact.getValue2() ) );
      LogInfo.logs( "P123 error: " + MatrixOps.diff(moments_.getValue3(), momentsExact.getValue3() ) );
    }
    return recoverHMM(K, tensorMethod.recoverParameters(K, moments_), smoothMeasurements);
  }

  static MixtureOfGaussians recoverGMM(int K, Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> measurements, double smoothMeasurements) {
    LogInfo.begin_track("cast-conditional-means-to-gmm");
    // Populate params
    SimpleMatrix pi = measurements.getValue0();
    // Ignoring M1 which is arbitrarily complicated.
    SimpleMatrix M1 = measurements.getValue1();
    SimpleMatrix M2 = measurements.getValue2();
    SimpleMatrix M3 = measurements.getValue3();
    int D = M1.numRows();

    pi = MatrixOps.projectOntoSimplex(pi.transpose(), smoothMeasurements).transpose();

    MixtureOfGaussians model = new MixtureOfGaussians( K, D, 3,
        pi, new SimpleMatrix[]{M1,M2,M3},
        new SimpleMatrix[][]{} ); // I know nothing about these covariances
    LogInfo.end_track("cast-conditional-means-to-hmm");

    return model;
  }
  public static MixtureOfGaussians recoverGMM(int K, ComputableMoments moments, double smoothMeasurements) {
    return recoverGMM(K, tensorMethod.randomizedRecoverParameters(K, moments), smoothMeasurements);
  }

  public static MixtureOfGaussians recoverGMM(int K, HasExactMoments moments, double smoothMeasurements) {
    return recoverGMM(K, tensorMethod.recoverParameters(K, moments.computeExactMoments()), smoothMeasurements);
  }

  public static MixtureOfGaussians recoverGMM(int K, int N, HasSampleMoments moments, double smoothMeasurements) {
    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> moments_ = moments.computeSampleMoments(N);
    if( moments instanceof HasExactMoments) {
      Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> momentsExact = ((HasExactMoments) moments).computeExactMoments();
      LogInfo.logs( "P13 error: " + MatrixOps.diff(moments_.getValue0(), momentsExact.getValue0() ) );
      LogInfo.logs( "P12 error: " + MatrixOps.diff(moments_.getValue1(), momentsExact.getValue1() ) );
      LogInfo.logs( "P32 error: " + MatrixOps.diff(moments_.getValue2(), momentsExact.getValue2() ) );
      LogInfo.logs( "P123 error: " + MatrixOps.diff(moments_.getValue3(), momentsExact.getValue3() ) );
    }
    return recoverGMM(K, tensorMethod.recoverParameters(K, moments_), smoothMeasurements);
  }

}
