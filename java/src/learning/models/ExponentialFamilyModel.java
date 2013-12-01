package learning.models;

import fig.basic.Indexer;
import learning.models.loglinear.Feature;
import learning.models.loglinear.ParamsVec;
import learning.utils.Counter;

import java.util.Random;

/**
 * Subclass of models of the form p(x,y) \propto \exp( \theta^T phi(x,y) ).
 */
public interface ExponentialFamilyModel<T> {
  public int getK();
  public int getD();
  public int numFeatures();
  public ParamsVec newParamsVec();
  public double getLogLikelihood(ParamsVec parameters);
  public double getLogLikelihood(ParamsVec parameters, T example);
  public double getLogLikelihood(ParamsVec parameters, Counter<T> examples);
  public ParamsVec getMarginals(ParamsVec parameters);
  public ParamsVec getMarginals(ParamsVec parameters, T example);
  public ParamsVec getMarginals(ParamsVec parameters, Counter<T> examples);
  public ParamsVec getSampleMarginals(Counter<T> examples);

  public T drawSample(ParamsVec parameters, Random genRandom);
  public Counter<T> drawSamples(ParamsVec parameters, Random genRandom, int n);

}

