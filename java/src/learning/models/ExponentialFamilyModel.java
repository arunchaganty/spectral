package learning.models;

import fig.basic.Indexer;
import learning.models.loglinear.Example;
import learning.models.loglinear.Feature;
import learning.models.loglinear.ParamsVec;
import learning.utils.Counter;

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

}

