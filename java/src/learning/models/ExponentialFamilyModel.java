package learning.models;

import learning.linalg.FullTensor;
import learning.linalg.MatrixOps;
import learning.common.Counter;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

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

  abstract protected void updateMarginals(Params parameters, T example, double scale, double count, Params marginals);
  abstract protected void updateMarginals(Params parameters, int L, double scale, double count, Params marginals);

  /**
   * Post processing steps, if required
   * @param parameters
   * @param marginals
   */
  protected void postUpdateMarginals(Params parameters, Params marginals) {
  }
  public void updateMarginals(Params parameters, Counter<T> examples, double scale, Params marginals) {
    double cnt = 0.;
    for(T example : examples) {
      cnt += examples.getFraction(example);
      updateMarginals(parameters, example, scale * examples.getFraction(example), cnt, marginals);
    }
    postUpdateMarginals(parameters, marginals);
  }
  public void updateMarginals(Params parameters, int[] histogram, double scale, Params marginals) {
    double sum = MatrixOps.sum(histogram);
    double cnt = 0.;
    for(int length = 0; length < histogram.length; length++) {
      if(histogram[length] > 0.) {
        cnt += histogram[length]/sum;
        updateMarginals(parameters, length, scale * histogram[length]/sum, cnt, marginals);
      }
    }
    postUpdateMarginals(parameters, marginals);
  }

  public Params getMarginals(Params parameters) {
    return getMarginals(parameters, (T) null);
  }
  public Params getMarginals(Params parameters, T example) {
    Counter<T> examples = new Counter<>();
    examples.add(example);
    return getMarginals(parameters, examples);
  }

  public Params getMarginals(Params parameters, Counter<T> examples) {
    Params marginals = newParams();
    updateMarginals(parameters, examples, 1.0, marginals);
    return marginals;
  }
  public Params getSampleMarginals(Counter<T> examples) {
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

  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> getMoments(Counter<T> data) {
    int D = getD();
    SimpleMatrix P12 = new SimpleMatrix(D,D);
    SimpleMatrix P13 = new SimpleMatrix(D,D);
    SimpleMatrix P32 = new SimpleMatrix(D,D);
    FullTensor P123 = new FullTensor(D,D,D);

    double count = 0.;

    // Iterate over data and compute.
    for(T ex : data) {
      count += updateMoments(ex, data.getFraction(ex), P12, P13, P32, P123);
    }
    // Rescale everything by count (which is usually 1)
    P12.scale(1./count);
    P13.scale(1./count);
    P32.scale(1./count);
    P123.scale(1./count);

    return Quartet.with(P12, P13, P32, P123);
  }
  public Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> getMoments(Params params) {
    int D = getD();
    SimpleMatrix P12 = new SimpleMatrix(D,D);
    SimpleMatrix P13 = new SimpleMatrix(D,D);
    SimpleMatrix P32 = new SimpleMatrix(D,D);
    FullTensor P123 = new FullTensor(D,D,D);
    updateMoments(params, 1.0, P12, P13, P32, P123);

    return Quartet.with(P12, P13, P32, P123);
  }


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

  public double updateMoments(Params params, double scale, SimpleMatrix P12, SimpleMatrix P13, SimpleMatrix P32, FullTensor P123) {
    throw new RuntimeException("not supported");
  }

  public Params recoverFromMoments(Counter<T> data, SimpleMatrix pi, SimpleMatrix M1, SimpleMatrix M2, SimpleMatrix M3, double smoothMeasurements) {
    throw new RuntimeException("not supported");
  }

  public abstract int getSize(T example);
}

