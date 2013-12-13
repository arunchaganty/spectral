package learning.models;

import fig.basic.Indexer;
import learning.linalg.FullTensor;
import learning.models.loglinear.Example;
import learning.models.loglinear.Feature;
import learning.models.loglinear.ParamsVec;
import learning.utils.Counter;
import org.ejml.simple.SimpleMatrix;

import java.util.Random;

/**
 * Subclass of models of the form p(x,y) \propto \exp( \theta^T phi(x,y) ).
 */
public abstract class ExponentialFamilyModel<T> {
  abstract public int getK();
  abstract public int getD();
  abstract public int numFeatures();
  abstract public ParamsVec newParamsVec();
  abstract public double getLogLikelihood(ParamsVec parameters);
  abstract public double getLogLikelihood(ParamsVec parameters, T example);
  abstract public double getLogLikelihood(ParamsVec parameters, Counter<T> examples);

  public double getProbability(ParamsVec parameters, T ex) {
    return Math.exp( getLogLikelihood(parameters,ex) - getLogLikelihood(parameters));
  }

  abstract public ParamsVec getMarginals(ParamsVec parameters);
  abstract public ParamsVec getMarginals(ParamsVec parameters, T example);
  abstract public ParamsVec getMarginals(ParamsVec parameters, Counter<T> examples);
  public ParamsVec getSampleMarginals(Counter<Example> examples) {
    throw new RuntimeException();
  }
  public Counter<Example> getDistribution(ParamsVec params) {
    throw new RuntimeException();
  }

  abstract public Counter<T> drawSamples(ParamsVec parameters, Random genRandom, int n);
  public T drawSample(ParamsVec parameters, Random rnd) {
    return drawSamples(parameters, rnd, 1).iterator().next();
  }

  /**
   * Update the moments
   * @param ex
   * @param count
   * @param P12
   * @param P13
   * @param P32
   * @param P123
   * @return - the number of updates made
   */
  public abstract double updateMoments(T ex, double count, SimpleMatrix P12, SimpleMatrix P13, SimpleMatrix P32, FullTensor P123);

}

