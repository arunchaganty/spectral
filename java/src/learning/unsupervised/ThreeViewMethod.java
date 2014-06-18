package learning.unsupervised;

import fig.basic.*;
import learning.linalg.FullTensor;
import learning.models.ExponentialFamilyModel;
import learning.models.Params;
import learning.common.Counter;
import learning.spectral.TensorMethod;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

/**
 * Expectation Maximization for a model
 */
public class ThreeViewMethod<T> {
  @OptionSet(name = "TM")
  public TensorMethod algo = new TensorMethod();

  public Params solve(
          ExponentialFamilyModel<T> modelA,
          Counter<T> data,
          Params initParams,
          double smoothMeasurements
  ) {
    LogInfo.begin_track("solveMoM");

    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> moments_ = modelA.getMoments(initParams);
    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> moments = modelA.getMoments(data);
    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, SimpleMatrix> factorization = algo.recoverParameters(modelA.getK(), moments);

    Params theta = modelA.recoverFromMoments(
            data,
            factorization.getValue0(),
            factorization.getValue1(),
            factorization.getValue2(),
            factorization.getValue3(),
            smoothMeasurements
    );

    return theta;
  }
}
