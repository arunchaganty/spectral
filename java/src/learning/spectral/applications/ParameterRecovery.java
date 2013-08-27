package learning.spectral.applications;

import fig.basic.LogInfo;
import learning.data.ComputableMoments;
import learning.data.HasExactMoments;
import learning.data.HasSampleMoments;
import learning.linalg.MatrixFactory;
import learning.linalg.MatrixOps;
import learning.models.HiddenMarkovModel;
import learning.spectral.TensorMethod;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

/**
 * Canoncial class for parameter recovery using the "spectral" tensor factorization method.
 */
public class ParameterRecovery {
  static TensorMethod tensorMethod = new TensorMethod();


  static HiddenMarkovModel recoverHMM(int K, Quartet<SimpleMatrix,SimpleMatrix,SimpleMatrix,SimpleMatrix> measurements, double smoothMeasurements) {
    LogInfo.begin_track("cast-conditional-means-to-hmm");
    // Populate params
    // construct O T and pi
    SimpleMatrix O = measurements.getValue2();
    SimpleMatrix T = O.pseudoInverse().mult( measurements.getValue3() );

    // Initialize pi to be random.
    SimpleMatrix pi = measurements.getValue0(); // This is just a random guess.

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
    return recoverHMM(K, tensorMethod.recoverParameters(K, moments.computeSampleMoments(N)), smoothMeasurements);
  }
}
