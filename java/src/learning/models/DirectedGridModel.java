package learning.models;

import fig.basic.Indexer;
import fig.prob.Multinomial;
import learning.common.Counter;
import learning.common.Utils;
import learning.linalg.FullTensor;
import learning.linalg.MatrixOps;
import learning.linalg.RandomFactory;
import learning.models.BasicParams;
import learning.models.ExponentialFamilyModel;
import learning.models.Params;
import learning.models.loglinear.*;
import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

/**
 * Directed
 */
public class DirectedGridModel extends ExponentialFamilyModel<Example> {
  final int K; final int D; final int L;
  final int rows; final int cols;
  final Indexer<Feature> indexer;

  public int pi(int h) {
    return h;
  }
  public int t(int h_, int h) {
    return h_ * K + h + K;
  }
  public int tC(int h_u, int h_l, int h) {
    return h_u * K * K + h_l * K + h + K * K + K;
  }
  public int o(int h, int x) {
    return h * D + x + K*K*K + K*K + K;
  }

  public int getL() { return L; }

  public Feature piFeature(int h) {
    return new UnaryFeature(h, "pi");
  }
  public Feature tFeature(int h_, int h) {
    return new BinaryFeature(h_, h);
  }
  public Feature tCFeature(int h_u, int h_l, int h) {
    return new TernaryFeature(h_u, h_l, h);
  }
  public Feature oFeature(int h, int x) {
    return new UnaryFeature(h, "x="+x);
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
      // Normalize
      normalize();

    }

    public boolean isValid() {
      // - pi
      {
        double z = 0.;
        for(int h = 0; h < K; h++) z += Math.abs(weights[pi(h)]);
        if(!MatrixOps.equal(z, 1.0)) return false;
      }

      // - T
      for(int h1 = 0; h1 < K; h1++) {
        double z = 0.;
        for(int h2 = 0; h2 < K; h2++) z += Math.abs(weights[t(h1,h2)]);
        if(!MatrixOps.equal(z, 1.0)) return false;
      }

      // - TC
      for(int h1 = 0; h1 < K; h1++) {
        for(int h2 = 0; h2 < K; h2++) {
          double z = 0.;
          for(int h3 = 0; h3 < K; h3++) z += Math.abs(weights[tC(h1,h2,h3)]);
          if(!MatrixOps.equal(z, 1.0)) return false;
        }
      }

      // - O
      for(int h1 = 0; h1 < K; h1++) {
        double z = 0.;
        for(int x = 0; x < D; x++) z += Math.abs(weights[o(h1,x)]);
        if(!MatrixOps.equal(z, 1.0)) return false;
      }
      return true;
    }

    public double[] getPi() {
      double[] pi = new double[K];
      for(int h = 0; h < K; h++)
        pi[h] = weights[pi(h)];
      return pi;
    }

    public double[][] getT() {
      double[][] T = new double[K][K];
      for(int h1 = 0; h1 < K; h1++)
        for(int h2 = 0; h2 < K; h2++)
          T[h1][h2] = weights[t(h1,h2)];
      return T;
    }

    public double[][][] getTC() {
      double[][][] T = new double[K][K][K];
      for(int h1 = 0; h1 < K; h1++)
        for(int h2 = 0; h2 < K; h2++)
          for(int h3 = 0; h3 < K; h3++)
            T[h1][h2][h3] = weights[tC(h1, h2, h3)];
      return T;
    }

    public double[][] getO() {
      double[][] O = new double[K][D];
      for(int h1 = 0; h1 < K; h1++)
        for(int x = 0; x < D; x++)
          O[h1][x] = weights[o(h1, x)];
      return O;
    }

    public void normalize() {
      {
        double z = 0.;
        for(int h = 0; h < K; h++) z += Math.abs(weights[pi(h)]);
        for(int h = 0; h < K; h++) weights[pi(h)] = Math.abs(weights[pi(h)])/z;
      }

      // - T
      for(int h1 = 0; h1 < K; h1++) {
        double z = 0.;
        for(int h2 = 0; h2 < K; h2++) z += Math.abs(weights[t(h1,h2)]);
        for(int h2 = 0; h2 < K; h2++) weights[t(h1,h2)] = Math.abs(weights[t(h1,h2)])/z;
      }

      // - TC
      for(int h1 = 0; h1 < K; h1++) {
        for(int h2 = 0; h2 < K; h2++) {
          double z = 0.;
          for(int h3 = 0; h3 < K; h3++) z += Math.abs(weights[tC(h1,h2,h3)]);
          for(int h3 = 0; h3 < K; h3++) weights[tC(h1,h2,h3)] = Math.abs(weights[tC(h1,h2,h3)])/z;
        }
      }

      // - O
      for(int h1 = 0; h1 < K; h1++) {
        double z = 0.;
        for(int x = 0; x < D; x++) z += Math.abs(weights[o(h1,x)]);
        for(int x = 0; x < D; x++) weights[o(h1, x)] = Math.abs(weights[o(h1, x)])/z;
      }
      assert isValid();
    }
  }

  protected class IntermediateState {
    int stack = 0;
    boolean use = false;
    Double logZ;

    void start() {
      if(stack == 0) {
        use = true;
        logZ = null;
      }
      stack++;
    }
    void stop() {
      stack--;
      if(stack == 0) {
        use = false;
        logZ = null;
      }
    }
  }
  IntermediateState intermediateState = new IntermediateState();

  public DirectedGridModel(int K, int D, int L) {
    assert L == 4;
    this.K = K;
    this.D = D;
    this.L = L;
    this.rows = L/2;
    this.cols = 2;

    // Populate indexer
    indexer = new Indexer<>();
    for(int h = 0; h < K; h++) {
      assert indexer.getIndex(piFeature(h)) == pi(h);
    }
    for(int h_ = 0; h_ < K; h_++)
      for(int h = 0; h < K; h++)
        assert indexer.getIndex(tFeature(h_, h)) == t(h_, h);
    for(int h_u = 0; h_u < K; h_u++)
      for(int h_l = 0; h_l < K; h_l++)
          for(int h = 0; h < K; h++)
            assert indexer.getIndex(tCFeature(h_u, h_l, h)) == tC(h_u, h_l, h);
    for(int h = 0; h < K; h++)
      for(int x = 0; x < D; x++)
        assert indexer.getIndex(oFeature(h, x)) == o(h,x);
    indexer.lock();

    // Oh look, I'm going to enumerate the whole damn thing and save a few hours. Aren't I clever.
  }

  @Override
  public int getK() {
    return K;
  }

  @Override
  public int getD() {
    return D;
  }

  @Override
  public int numFeatures() {
    // K*K for transitions and K*D for emissions.
    return K * D + K * K * K + K * K + K;
  }

  @Override
  public Parameters newParams() {
    return new Parameters(K, indexer);
  }

  public int hIdx(int r, int c) {
    return r * (cols) + c;
  }
  public int oIdx(int r, int c, int d) {
    return r * (cols * 2) + c * (2) + d;
  }

  @Override
  public double getLogLikelihood(Params parameters, int L) {
    return 0.0;
  }

  @Override
  public double getLogLikelihood(Params parameters, Example example) {
    throw new RuntimeException("Not yet implemented.");
//    if(example == null) return getLogLikelihood(parameters,L);
//    assert example.x.length == 2 * this.L;
//    double lhood = Double.NEGATIVE_INFINITY;
//    double[] weights = parameters.toArray();
//    int[] h = example.h;
    // Iterate through and add the cost of everything.
//    for(int[] hidden : hiddenConfigurations) {
//      example.h = hidden;
//      double lhood_ = getFullLikelihood(parameters, example);
//      lhood = MatrixOps.logsumexp(lhood, lhood_);
//    }
//    example.h = h;
//
//    return lhood;
  }

  public double getFullLikelihood(Params parameters, Example example) {
    assert example.x.length == 2 * this.L;
    assert example.h != null;

    double[] weights = parameters.toArray();
    // Iterate through and add the cost of everything.
    int[] hidden = example.h;
    double lhood = 0.;
     // - O
    for(int row = 0; row < rows; row++) {
      for(int col = 0; col < cols; col++) {
        // Add observations.
        int h = hidden[hIdx(row,col)];
        int x1 = example.x[oIdx(row, col, 0)];
        int x2 = example.x[oIdx(row,col,1)];
        lhood += weights[o(h, x1)]; // For d = 1, d = 2
        lhood += weights[o(h, x2)]; // For d = 1, d = 2
      }
    }

    // - pi
    {
      int h = hidden[hIdx(0,0)];
      lhood += weights[pi(h)];
    }

    // - T
    for(int row = 0; row < rows; row++) {
      for(int col = 0; col < cols; col++) {
        // Add contribution from parents
        int h = hidden[hIdx(row,col)];
        if( row > 0 && col > 0) {
          int h_u = hidden[hIdx(row-1,col)];
          int h_l = hidden[hIdx(row,col-1)];
          lhood += weights[tC(h_u,h_l,h)];
        }
        else if(row > 0) {
          int h_u = hidden[hIdx(row-1,col)];
          lhood += weights[t(h_u,h)];
        }
        if(col > 0) {
          int h_l = hidden[hIdx(row,col-1)];
          lhood += weights[t(h_l,h)];
        }
      }
    }

    return lhood;
  }
  public double getFullProbability(Params parameters, Example example) {
    return Math.exp(getFullLikelihood(parameters,example));
  }


  @Override
  public void updateMarginals(Params parameters, Example example, double scale, double count, Params marginals) {
    throw new RuntimeException("not yet implemented.");
//    assert example == null || example.x.length == 2 * this.L;
//
//    intermediateState.start();
//    // Because we mutate example
//    double logZ = (example == null)
//            ? getLogLikelihood(parameters, L)
//            : getLogLikelihood(parameters, example);
//    Example example_ = new Example();
//    for(int[] observed : (example == null)
//            ? Arrays.asList(observedConfigurations)
//            : Collections.singleton(example.x)) {
//      // Iterate through and add the cost of everything.
//      example_.x = observed;
//      for(int[] hidden : hiddenConfigurations) {
//        example_.h = hidden;
//        double pr = Math.exp(getFullLikelihood(parameters, example_) - logZ);
//        updateFullMarginals(example_, scale * pr, marginals);
//      }
//    }
//    intermediateState.stop();
  }

  public void updateFullMarginals(Example ex, double scale, Params marginals) {
    // Because we mutate example
    double[] marginals_ = marginals.toArray();
    // Iterate through and add the cost of everything.
    {
      int h = ex.h[hIdx(0,0)];
      marginals_[pi(h)] += scale;
    }
    for(int row = 0; row < rows; row++) {
      for(int col = 0; col < cols; col++) {
        // Add observations.
        int h = ex.h[hIdx(row,col)];
        int x1 = ex.x[oIdx(row, col, 0)];
        int x2 = ex.x[oIdx(row,col,1)];
        marginals_[o(h,x1)] += scale;
        marginals_[o(h,x2)] += scale;
      }
    }
    for(int row = 0; row < rows; row++) {
      for(int col = 0; col < cols; col++) {
        int h = ex.h[hIdx(row,col)];
        // Add transitions.
        if(row == 0 && col == 0) {
        } else if(row == 0) {
          int h_l = ex.h[hIdx(row,col-1)];
          marginals_[t(h_l,h)] += scale;
        }
        else if(col == 0) {
          int h_u = ex.h[hIdx(row-1,col)];
          marginals_[t(h_u,h)] += scale;
        }
        else {
          int h_u = ex.h[hIdx(row-1,col)];
          int h_l = ex.h[hIdx(row,col-1)];
          marginals_[tC(h_u,h_l,h)] += scale;
        }
      }
    }
  }

  @Override
  public void updateMarginals(Params parameters, int L, double scale, double count, Params marginals) {
    assert( this.L == L );
    updateMarginals(parameters, (Example)null, scale, count, marginals);
  }

  @Override
  public Counter<Example> drawSamples(Params params_, Random genRandom, int n) {
    if(!(params_ instanceof Parameters))
      throw new IllegalArgumentException();
    Parameters params = (Parameters) params_;
    Counter<Example> examples = new Counter<>();

    double[] pi = params.getPi();
    double[][] T = params.getT();
    double[][][] TC = params.getTC();
    double[][] O = params.getO();


    for(int i = 0; i < n; i++) {
      examples.add(drawSample(params, genRandom, pi, T, TC, O));
    }
    return examples;
  }
  public Example drawSample(Params parameters, Random genRandom, double[] pi, double[][] T, double[][][] TC, double[][] O) {
    Example ex = new Example( new int[L*2], new int[L] );

    // Sample h first.
    {
      ex.h[hIdx(0,0)] = RandomFactory.multinomial(genRandom, pi);
      for(int row = 0; row < rows; row++) { // row
        for(int col = 0; col < cols; col++) { // col
          if(row == 0 && col == 0) {
          }
          else if(row == 0) {
            int h_l = ex.h[hIdx(row,col-1)];
            ex.h[hIdx(row,col)] = RandomFactory.multinomial(genRandom, T[h_l]);
          }
          else if(col == 0) {
            int h_u = ex.h[hIdx(row-1,col)];
            ex.h[hIdx(row,col)] = RandomFactory.multinomial(genRandom, T[h_u]);
          }
          else {
            int h_u = ex.h[hIdx(row-1,col)];
            int h_l = ex.h[hIdx(row,col-1)];
            ex.h[hIdx(row,col)] = RandomFactory.multinomial(genRandom, TC[h_u][h_l]);
          }
        }
      }
    }

    {
      for(int row = 0; row < rows; row++) { // row
        for(int col = 0; col < cols; col++) { // col
          int h = ex.h[hIdx(row,col)];
          ex.x[oIdx(row,col,0)] = RandomFactory.multinomial(genRandom, O[h]);
          ex.x[oIdx(row,col,1)] = RandomFactory.multinomial(genRandom, O[h]);
        }
      }
    }

    return ex;
  }

  @Override
  public double updateMoments(Example ex, double count, SimpleMatrix P12, SimpleMatrix P13, SimpleMatrix P32, FullTensor P123) {
    double updates = 0;
    for( int r = 0; r < rows; r++ ) {
      for( int c = 0; c < cols; c++ ) {
        int x2 = ex.x[oIdx(r,c,0)];
        int x3 = ex.x[oIdx(r,c,1)]; // Just because we extract M3 - so lets pivot around this.
        // TODO: Average over all the other possibilities
        int x1 = ex.x[oIdx( (r+1) % rows,c,0)];
        P13.set( x1, x3, P13.get( x1, x3 ) + count);
        P12.set( x1, x2, P12.get( x1, x2 ) + count);
        P32.set( x3, x2, P32.get( x3, x2 ) + count);
        P123.set( x1, x2, x3, P123.get(x1, x2, x3) + count );
        updates += count;
      }
    }
    return updates;
  }

  @Override
  public Params getSampleMarginals(Counter<Example> examples) {
    Parameters marginals = newParams();
    for(Example ex : examples) {
      updateFullMarginals(ex, examples.getFraction(ex), marginals);
    }
    // Normalize marginals
    marginals.normalize();


    return marginals;
  }

  public Counter<Example> getDistribution(Params params) {
    Counter<Example> examples = new Counter<>();
    for(int[] hidden : Utils.enumerate(K, L)) {
      for(int[] observed : Utils.enumerate(D, 2*L)) {
        Example ex = new Example(observed, hidden);
        examples.set(ex, getFullProbability(params, ex));
      }
    }
    intermediateState.stop();

    return examples;
  }

  public int getSize(Example ex) {
    return ex.x.length;
  }


}
