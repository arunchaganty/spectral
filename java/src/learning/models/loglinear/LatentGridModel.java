package learning.models.loglinear;

import fig.basic.Indexer;
import fig.prob.Multinomial;
import learning.common.Counter;
import learning.common.Utils;
import learning.linalg.FullTensor;
import learning.linalg.MatrixOps;
import learning.models.BasicParams;
import learning.models.ExponentialFamilyModel;
import learning.models.Params;
import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

/**
 * A stupid implementation of the Grid model
 */
public class LatentGridModel extends ExponentialFamilyModel<Example> {
  final int K; final int D; final int L;
  final int rows; final int cols;
  final Indexer<Feature> indexer;
  final int[][] hiddenConfigurations;
  final int[][] observedConfigurations;

  public int o(int h, int x) {
    return h * D + x;
  }
  public int t(int h_, int h) {
    return h_ * K + h + K * D;
  }

  public int getL() { return L; }

  public Feature oFeature(int h, int x) {
    return new UnaryFeature(h, "x="+x);
  }
  public Feature tFeature(int h_, int h) {
    return new BinaryFeature(h_, h);
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

  public LatentGridModel(int K, int D, int L) {
    assert L == 4;
    this.K = K;
    this.D = D;
    this.L = L;
    this.rows = L/2;
    this.cols = 2;

    // Populate indexer
    indexer = new Indexer<>();
    for(int h = 0; h < K; h++)
      for(int x = 0; x < D; x++)
        indexer.getIndex(oFeature(h, x));
    for(int h_ = 0; h_ < K; h_++)
      for(int h = 0; h < K; h++)
        indexer.getIndex(tFeature(h_, h));
    indexer.lock();

    // Oh look, I'm going to enumerate the whole damn thing and save a few hours. Aren't I clever.
    hiddenConfigurations = Utils.enumerate(K,L);
    observedConfigurations = Utils.enumerate(D,2*L);
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
    return K * K + K * D;
  }

  @Override
  public Params newParams() {
    return new BasicParams(K, indexer);
  }

  public int hIdx(int r, int c) {
    return r * (cols) + c;
  }
  public int oIdx(int r, int c, int d) {
    return r * (cols * 2) + c * (2) + d;
  }

  @Override
  public double getLogLikelihood(Params parameters, int L) {
    if(intermediateState.use && intermediateState.logZ != null) return intermediateState.logZ;
    double lhood = Double.NEGATIVE_INFINITY;
    Example ex = new Example();
    for(int[] x : observedConfigurations)  {
      ex.x = x;
      lhood = MatrixOps.logsumexp(lhood,getLogLikelihood(parameters, ex));
    }
    if(intermediateState.use) intermediateState.logZ = lhood;
    return lhood;
  }

  @Override
  public double getLogLikelihood(Params parameters, Example example) {
    if(example == null) return getLogLikelihood(parameters,L);
    assert example.x.length == 2 * this.L;
    double lhood = Double.NEGATIVE_INFINITY;
    double[] weights = parameters.toArray();
    int[] h = example.h;
    // Iterate through and add the cost of everything.
    for(int[] hidden : hiddenConfigurations) {
      example.h = hidden;
      double lhood_ = getFullLikelihood(parameters, example);
      lhood = MatrixOps.logsumexp(lhood, lhood_);
    }
    example.h = h;

    return lhood;
  }

  public double getFullLikelihood(Params parameters, Example example) {
    assert example.x.length == 2 * this.L;
    assert example.h != null;

    double[] weights = parameters.toArray();
    // Iterate through and add the cost of everything.
    int[] hidden = example.h;
    double lhood = 0.;
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
    for(int row = 0; row < rows; row++) {
      for(int col = 0; col < cols; col++) {
        // Add observations.
        int h_ = hidden[hIdx(row,col)];
        if(row < rows-1) {
          int h = hidden[hIdx(row+1,col)];
          lhood += weights[t(h_,h)];
        }
        if(col < cols-1) {
          int h = hidden[hIdx(row,col+1)];
          lhood += weights[t(h_,h)];
        }
      }
    }

    return lhood;
  }
  public double getFullProbability(Params parameters, Example example) {
    return Math.exp(getFullLikelihood(parameters,example) - getLogLikelihood(parameters,L));
  }


  @Override
  public void updateMarginals(Params parameters, Example example, double scale, double count, Params marginals) {
    assert example == null || example.x.length == 2 * this.L;

    intermediateState.start();
    // Because we mutate example
    double logZ = (example == null)
            ? getLogLikelihood(parameters, L)
            : getLogLikelihood(parameters, example);
    Example example_ = new Example();
    for(int[] observed : (example == null)
            ? Arrays.asList(observedConfigurations)
            : Collections.singleton(example.x)) {
      // Iterate through and add the cost of everything.
      example_.x = observed;
      for(int[] hidden : hiddenConfigurations) {
        example_.h = hidden;
        double pr = Math.exp(getFullLikelihood(parameters, example_) - logZ);
        updateFullMarginals(example_, scale * pr, marginals);
      }
    }
    intermediateState.stop();
  }

  public void updateFullMarginals(Example example, double scale, Params marginals) {
    // Because we mutate example
    double[] marginals_ = marginals.toArray();
    // Iterate through and add the cost of everything.
    for(int row = 0; row < rows; row++) {
      for(int col = 0; col < cols; col++) {
        // Add observations.
        int h = example.h[hIdx(row,col)];
        int x1 = example.x[oIdx(row, col, 0)];
        int x2 = example.x[oIdx(row,col,1)];
        marginals_[o(h,x1)] += scale;
        marginals_[o(h,x2)] += scale;
      }
    }
    for(int row = 0; row < rows; row++) {
      for(int col = 0; col < cols; col++) {
        // Add observations.
        int h_ = example.h[hIdx(row,col)];
        if(row < rows-1) {
          int h = example.h[hIdx(row+1,col)];
          marginals_[t(h_,h)] += scale;
        }
        if(col < cols-1) {
          int h = example.h[hIdx(row,col+1)];
          marginals_[t(h_,h)] += scale;
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
  public Counter<Example> drawSamples(Params parameters, Random genRandom, int n) {
    intermediateState.start();

    double[] multinomial = new double[hiddenConfigurations.length * observedConfigurations.length];
    Example ex = new Example();
    double logZ = getLogLikelihood(parameters, L);
    for(int i = 0; i < hiddenConfigurations.length; i++) {
      ex.h = hiddenConfigurations[i];
      for(int j = 0; j < observedConfigurations.length; j++) {
        ex.x = observedConfigurations[j];
        multinomial[i*observedConfigurations.length + j] = Math.exp(getFullLikelihood(parameters, ex) - logZ);
      }
    }
    // There is some times some numerical error...
    MatrixOps.scale(multinomial, 1. / MatrixOps.sum(multinomial));

    Counter<Example> examples = new Counter<>();
    for(int i = 0; i < n; i++) {
      int choice = Multinomial.sample(genRandom, multinomial);
      int[] x = observedConfigurations[choice % observedConfigurations.length];
      int[] h = hiddenConfigurations[choice / observedConfigurations.length];
      examples.add(new Example(x,h));
    }

    intermediateState.stop();
    return examples;
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
    Params marginals = newParams();
    for(Example ex : examples) {
      updateFullMarginals(ex, examples.getFraction(ex), marginals);
    }
    return marginals;
  }

  public Counter<Example> getDistribution(Params params) {
    Counter<Example> examples = new Counter<>();
    intermediateState.start();
    for(int[] hidden : hiddenConfigurations) {
      for(int[] observed : observedConfigurations) {
        Example ex = new Example(observed, hidden);
        examples.set(ex, getProbability(params, ex));
      }
    }
    intermediateState.stop();

    return examples;
  }

  public int getSize(Example ex) {
    return ex.x.length;
  }


}
