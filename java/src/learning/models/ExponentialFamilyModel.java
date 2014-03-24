package learning.models;

import learning.linalg.FullTensor;
import learning.linalg.MatrixOps;
import learning.models.loglinear.Example;
import learning.common.Counter;
import org.ejml.simple.SimpleMatrix;

import java.util.Random;

/**
 * Subclass of models of the form p(x,y) \propto \exp( \theta^T phi(x,y) ).
 */
public abstract class ExponentialFamilyModel<T> {
  abstract public int getK();
  abstract public int getD();
  abstract public int numFeatures();
  abstract public Params newParams();
  abstract public double getLogLikelihood(Params parameters, int L);
  abstract public double getLogLikelihood(Params parameters, T example);

  public double getLogLikelihood(Params parameters) {
    return getLogLikelihood(parameters, (T) null);
  }
  public double getLogLikelihood(Params parameters, int[] histogram) {
    double sum = MatrixOps.sum(histogram);
    double lhood = 0.;
    for(int length = 0; length < histogram.length; length++) {
      if(histogram[length] > 0.)
        lhood += histogram[length]/sum * getLogLikelihood(parameters, length);
    }
    return lhood;
  }
  public double getLogLikelihood(Params parameters, Counter<T> examples) {
    double lhood = 0.;
    for(T ex : examples) {
      lhood += examples.getFraction(ex) * getLogLikelihood(parameters, ex);
    }
    return lhood;
  }

  public double getProbability(Params parameters, T ex) {
    return Math.exp( getLogLikelihood(parameters,ex) - getLogLikelihood(parameters));
  }

  abstract public void updateMarginals(Params parameters, T example, double scale, Params marginals);
  abstract public void updateMarginals(Params parameters, int L, double scale, Params marginals);

  public void updateMarginals(Params parameters, Counter<T> examples, double scale, Params marginals) {
    for(T example : examples) {
      updateMarginals(parameters, example, scale * examples.getFraction(example), marginals);
//      updateMarginals(parameters, example, getProbability(parameters, example), marginals);
    }
  }
  public void updateMarginals(Params parameters, int[] histogram, double scale, Params marginals) {
    double sum = MatrixOps.sum(histogram);
    for(int length = 0; length < histogram.length; length++) {
      if(histogram[length] > 0.)
        updateMarginals(parameters, length, scale * histogram[length]/sum, marginals);
    }
  }


  public Params getMarginals(Params parameters) {
    Params marginals = newParams();
    updateMarginals(parameters, (T) null, 1.0, marginals);
    return marginals;
  }
  public Params getMarginals(Params parameters, T example) {
    Params marginals = newParams();
    updateMarginals(parameters, example, 1.0, marginals);
    return marginals;
  }
  public Params getMarginals(Params parameters, Counter<T> examples) {
    Params marginals = newParams();
    updateMarginals(parameters, examples, 1.0, marginals);
    return marginals;
  }
  public Params getSampleMarginals(Counter<Example> examples) {
    throw new RuntimeException();
  }
  public Counter<T> getDistribution(Params params) {
    throw new RuntimeException();
  }

  abstract public Counter<T> drawSamples(Params parameters, Random genRandom, int n);
  public T drawSample(Params parameters, Random rnd) {
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

  public T bestLabelling(Params params, T ex) {
    throw new RuntimeException("not supported");
  }

  /**
   * Get Hessian of the likelihood.
   * @param params
   * @return
   */
  public SimpleMatrix getHessian(Params params) {
    final double eps = 1e-6;
    double[] weights = params.toArray();
    double[] original = weights.clone();
    double[][] H = new double[weights.length][weights.length];

    for(int i = 0; i < weights.length; i++) {
      for(int j = 0; j < weights.length; j++) {
        weights[i] += eps;
        weights[j] += eps;
        double lhoodPlusPlus = getLogLikelihood(params);

        weights[j] -= 2*eps;
        double lhoodPlusMinus = getLogLikelihood(params);

        weights[i] -= 2*eps;
        double lhoodMinusMinus = getLogLikelihood(params);

        weights[j] += 2*eps;
        double lhoodMinusPlus =  getLogLikelihood(params);

        weights[i] += eps;
        weights[j] -= eps;

        H[i][j] = ((lhoodPlusPlus - lhoodMinusPlus)/(2*eps) + (lhoodMinusMinus  - lhoodPlusMinus)/(2*eps))/(2*eps);
      }
    }
    for(int i = 0; i < weights.length; i++) {
      assert MatrixOps.equal(original[i], weights[i]);
      weights[i] = original[i];
    }

    return new SimpleMatrix(H);
  }


}

