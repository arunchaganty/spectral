/**
 * learning.model
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.models;

import fig.basic.Indexer;
import fig.exec.Execution;
import learning.Misc;
import learning.common.Counter;
import learning.data.ComputableMoments;
import learning.data.HasExactMoments;
import learning.data.HasSampleMoments;
import learning.linalg.*;

import learning.models.loglinear.Feature;
import learning.models.loglinear.UnaryFeature;
import org.ejml.simple.SimpleMatrix;
import org.ejml.data.DenseMatrix64F;

import fig.basic.Option;
import fig.basic.OptionsParser;
import fig.basic.LogInfo;
import fig.prob.MultGaussian;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.IOException;

import java.util.Random;

import org.javatuples.*;

/**
 * A mixture of experts model
 */
public class MultiViewGaussian extends ExponentialFamilyModel<double[][]> {
  protected int K; // Number of components
  protected int D; // Dimensionality of space
  protected int L; // Number of views

  public int pi(int h) {
    return h;
  }
  // Middle K * L * D features are the means
  public int mu(int h, int view, int d) {
    return K + h * D * L + view * D + d;
  }

  // Last K features are variances
  // TODO: Support non-diagonal variances
  public int sigma(int h) {
    return K + K * D * L + h;
  }

  public class Parameters extends BasicParams {
    public Parameters(int K, Indexer<Feature> featureIndexer) {
      super(K, featureIndexer);
    }

    @Override
    public Parameters newParams() {
      return new Parameters(K, featureIndexer);
    }

    @Override
    public void initRandom(Random random, double noise) {
      super.initRandom(random, noise);
      // Now normalize appropriately
      normalize();
    }

    public void normalize() {
      // - pi
      {
        double z = 0.;
        for(int h = 0; h < K; h++) z += Math.abs(weights[pi(h)]);
        for(int h = 0; h < K; h++) weights[pi(h)] = Math.abs(weights[pi(h)])/z;
      }

      // - means (nothing)
      {
      }

      // - variances (are positive)
      for(int h = 0; h < K; h++) {
        weights[sigma(h)] = Math.abs(weights[sigma(h)]);
      }
    }

    public boolean isValid() {
      // - pi
      {
        double z = 0.;
        for(int h = 0; h < K; h++) z += Math.abs(weights[pi(h)]);
        if(!MatrixOps.equal(z,1.0)) return false;
      }

      // - means
      {
      }

      // - variances
      {
        for(int h = 0; h < K; h++) {
          if( weights[sigma(h)] < 0 ) return false;
        }
      }

      return true;
    }

    public double[] getPi() {
      double[] pi = new double[K];
      for(int h = 0; h < K; h++)
        pi[h] = weights[pi(h)];
      return pi;
    }
    public double[][][] getMeans() {
      double[][][] means = new double[K][L][D];
      for(int h = 0; h < K; h++)
        for(int v = 0; v < L; v++)
          for(int d = 0; d < D; d++)
            means[h][v][d] = weights[mu(h, v, d)];
      return means;
    }
    public double[] getSigma() {
      double[] Sigma = new double[K];
      for(int h1 = 0; h1 < K; h1++)
          Sigma[h1] = weights[sigma(h1)];
      return Sigma;
    }

    public Parameters with(double[] pi, double[][][] means, double[] sigma) {
      Parameters params = newParams();
      for(int h = 0; h < K; h++)
        params.weights[pi(h)] = pi[h];

      for(int h = 0; h < K; h++)
        for(int v = 0; v < L; v++)
          for(int d = 0; d < D; d++)
            params.weights[mu(h, v, d)] = means[h][v][d];

      for(int h = 0; h < K; h++)
        params.weights[sigma(h)] = sigma[h];

//      if(! params.isValid() )
//        LogInfo.warning("Invalid params; normalizing");
//      params.normalize();
      assert params.isValid();

      return params;
    }
  }

  public static Feature piFeature(int h) {
    return new UnaryFeature(h, "pi");
  }
  public static Feature meanFeature(int h, int view, int d) {
    return new UnaryFeature(h, "d_"+ view + "="+d);
  }
  public static Feature sigmaFeature(int h) {
    return new UnaryFeature(h, "sigma");
  }

  final Indexer<Feature> featureIndexer;
  public MultiViewGaussian(int K, int D, int L) {
    this.K = K;
    this.D = D;
    this.L = L;

    // Careful - this must follow the same ordering as the index numbers
    this.featureIndexer = new Indexer<>();
    for(int h = 0; h < K; h++) {
      featureIndexer.add(piFeature(h));
      assert featureIndexer.indexOf( piFeature(h) )  == pi(h);
    }
    for(int h = 0; h < K; h++) {
      for(int v = 0; v < L; v++) {
        for(int d = 0; d < D; d++) {
          featureIndexer.add(meanFeature(h, v, d));
          assert featureIndexer.indexOf( meanFeature(h, v, d) ) == mu(h, v, d);
        }
      }
    }
    for(int h = 0; h < K; h++) {
      featureIndexer.add(sigmaFeature(h));
      assert featureIndexer.indexOf(sigmaFeature(h)) == sigma(h);
    }
  }

  @Override
  public int getK() {
    return K;
  }

  @Override
  public int getD() {
    return D;
  }

  public int getL() {
    return L;
  }

  @Override
  public int numFeatures() {
    return K + K*L*D + K;
  }

  @Override
  public Parameters newParams() {
    return new Parameters(K, featureIndexer);
  }

  @Override
  public double getLogLikelihood(Params parameters, int L) {
    if(L != this.L)
      throw new RuntimeException("Invalid parameter L");
    else
      return 1.; // Directed model
  }

  /**
   * log-likelihood given a particular cluster
   */
  double lhood(Parameters parameters, int h, double[][] example) {

    double lhood = 0.;
    lhood += Math.log(parameters.weights[pi(h)]);
    for(int v = 0; v < L; v++) {
      for(int d = 0; d < K; d++ ) {
        if(example != null)
          lhood += - 0.5 * Math.pow(example[v][d] - parameters.weights[mu(h,v,d)], 2) / parameters.weights[sigma(h)];
      }
    }
    return lhood;
  }

  @Override
  public double getLogLikelihood(Params parameters_, double[][] example) {
    if(!(parameters_ instanceof Parameters))
      throw new IllegalArgumentException();
    Parameters parameters = (Parameters) parameters_;

    double lhood = Double.NEGATIVE_INFINITY;

    for(int h = 0; h < K; h++ ) {
      lhood = MatrixOps.logsumexp(lhood, lhood(parameters, h, example));
    }

    return lhood;
  }

  @Override
  public void updateMarginals(Params parameters_, double[][] example, double scale, double count, Params marginals_) {
    if(!(parameters_ instanceof Parameters))
      throw new IllegalArgumentException();
    if(!(marginals_ instanceof Parameters))
      throw new IllegalArgumentException();
    Parameters parameters = (Parameters) parameters_;
    Parameters marginals = (Parameters) marginals_;

    // Construct "responsibilities" and update marginals based on them
    double[] responsibilities = new double[K];
    for(int h = 0; h < K; h++) {
      responsibilities[h] = lhood(parameters, h, example);
    }
    double z = MatrixOps.logsumexp(responsibilities);
    for(int h = 0; h < K; h++) responsibilities[h] = Math.exp(responsibilities[h] - z);

    for(int h = 0; h < K; h++) {
      // - Update pi
      marginals.weights[pi(h)] += scale * (responsibilities[h] - marginals.weights[pi(h)]) / count;
    }

    assert MatrixOps.equal(MatrixOps.sum(marginals.getPi()), 1.0);

    // For online updates to other quantities weighted by \pi(h) * scale
    for(int h = 0; h < K; h++) {
      // - Update means
      for(int v = 0; v < L; v++) {
        for(int d = 0; d < K; d++ ) {
          marginals.weights[mu(h,v,d)] +=
                  scale * responsibilities[h] * (
                          ((example != null) ? example[v][d] : parameters.weights[mu(h,v,d)])
                           - marginals.weights[mu(h,v,d)]);
        }
      }
      // - Update sigma
      // TODO: Handle online updates (requires x^2 to be tracked)
      //      - Maybe keep track of some state?
//      for(int v = 0; v < L; v++) {
//        for(int d = 0; d < K; d++ ) {
          marginals.weights[sigma(h)] = 1.0;
//                  scale * responsibilities[h] / K * (Math.pow(example[v][d] - marginals.weights[mu(h,v,d)], 2) - marginals.weights[sigma(h)]);
//        }
//      }
    }
  }

  @Override
  public void updateMarginals(Params parameters_, int L, double scale, double count, Params marginals_) {
    // The parameters are the marginals!
    marginals_.plusEquals(scale, parameters_);
  }

  @Override
  public Counter<double[][]> drawSamples(Params parameters_, Random genRandom, int N) {
    if(!(parameters_ instanceof Parameters))
      throw new IllegalArgumentException();
    Parameters parameters = (Parameters) parameters_;

    Counter<double[][]> samples = new Counter<>();

    double[] pi = parameters.getPi();
    // Draw samples from N gaussians
    for(int n = 0; n < N; n++) {
      int k = RandomFactory.multinomial(genRandom, pi);
      double[][] point = new double[L][D];
      for(int v = 0; v < L; v++) {
        for(int d = 0; d < D; d++) {
          point[v][d] = parameters.weights[mu(k,v,d)] + RandomFactory.randn(genRandom, parameters.weights[sigma(k)]);
        }
      }

      samples.add(point);
    }

    return samples;
  }

  @Override
  public double updateMoments(double[][] ex, double scale, SimpleMatrix P12, SimpleMatrix P13, SimpleMatrix P32, FullTensor P123) {
    DenseMatrix64F P12_ = P12.getMatrix();
    DenseMatrix64F P13_ = P13.getMatrix();
    DenseMatrix64F P32_ = P32.getMatrix();
    for(int d1 = 0; d1 < D; d1++) {
      for(int d2 = 0; d2 < D; d2++) {
        P12_.add(d1, d2, scale * ex[0][d1] * ex[1][d2] );
        P13_.add(d1, d2, scale * ex[0][d1] * ex[2][d2] );
        P32_.add(d1, d2, scale * ex[2][d1] * ex[1][d2] );

        for(int d3 = 0; d3 < D; d3++) {
          P123.X[d1][d2][d3] += scale * ex[0][d1] * ex[1][d2] * ex[2][d3];
        }
      }
    }
    return scale; // Number of 'updates' made.
  }

  @Override
  public Params recoverFromMoments(SimpleMatrix pi, SimpleMatrix M1, SimpleMatrix M2, SimpleMatrix M3, double smoothMeasurements) {
    assert( M1.numRows() == D );
    assert( M1.numCols() == K );

    // Project onto the simplex and
    pi = MatrixOps.projectOntoSimplex(pi);

    LogInfo.log("pi: " + pi);
    LogInfo.log("M3: " + M3);

    Params params = newParams();
    for( int h = 0; h < K; h++ ) {
      params.toArray()[pi(h)] = pi.get(h);
      for( int d = 0; d < D; d++ ) {
        params.toArray()[mu(h,0,d)] = M1.get(h,d);
        params.toArray()[mu(h,1,d)] = M2.get(h,d);
        params.toArray()[mu(h,2,d)] = M3.get(h,d);
      }
    }
    return params;
  }

  @Override
  public int getSize(double[][] example) {
    return example.length;
  }
}

