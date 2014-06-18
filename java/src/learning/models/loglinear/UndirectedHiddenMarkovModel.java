package learning.models.loglinear;

import fig.basic.Fmt;
import fig.basic.Indexer;
import fig.basic.LogInfo;
import fig.basic.Pair;
import learning.linalg.FullTensor;
import learning.linalg.MatrixOps;
import learning.linalg.RandomFactory;
import learning.models.ExponentialFamilyModel;
import learning.models.Params;
import learning.common.Counter;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Undirected hidden Markov model
 */
public class UndirectedHiddenMarkovModel extends ExponentialFamilyModel<Example> {
  public int K;
  public int D;
  public int L;

  // TODO: Support initial state
  // First K*D features are observations
  public int o(int h, int x) {
    return h * D + x;
  }
  // Last K*K features are transitions
  public int t(int h_, int h) {
    return K * D + h_ * K + h;
  }

  public static class Parameters extends Params {
    final public int K;
    final public int D;
    final public double[] weights;
    final double[] expWeight;
    final Indexer<Feature> featureIndexer;

    public Parameters(int K, int D, Indexer<Feature> featureIndexer) {
      this.K = K;
      this.D = D;
      this.weights = new double[K*D + K*K];
      this.expWeight = new double[K*D + K*K];

      this.featureIndexer = featureIndexer;
    }

    @Override
    public Params newParams() {
      return new Parameters(K,D, featureIndexer);
    }

    @Override
    public Indexer<Feature> getFeatureIndexer() {
      return featureIndexer;
    }

    @Override
    public double[] toArray() {
      return weights;
    }

    boolean cacheValid = false;
    double[][][] cachedG;
    @Override
    public void cache() {
      if(cacheValid) return;
      precomputeG();
      cacheValid = true;
    }
    public boolean isCacheValid() {
      return cacheValid;
    }
    void precomputeG() {
      for(int i = 0; i < weights.length; i++)
        expWeight[i] = Math.exp(weights[i]);

      // Use a special index for t = 0 and for examples = 0
      Example ex = new Example(new int[]{0, 0});
      cachedG = new double[K+1][K][D+1];
      for(int y_ = -1; y_ < K; y_++) {
        for(int y = 0; y < K; y++) {
          for(int x = 0; x < D; x++) {
            if(y_ == -1) {
              ex.x[0] = x;
              cachedG[y_+1][y][x+1] = G__(0, y_, y, ex);
            } else {
              ex.x[1] = x;
              cachedG[y_+1][y][x+1] = G__(1, y_, y, ex);
            }
          }
          // Now compute the aggregate for 0
          cachedG[y_+1][y][0] = MatrixOps.sum(cachedG[y_+1][y]);
        }
      }
    }
    @Override
    public void invalidateCache() {
      cacheValid = false;
    }
    /**
     * Return \theta^T [\phi(y_{t-1}, y_{t}), \phi(y_{t}, x_{t}) ]
     */
    double G(int t, int y_, int y, Example ex) {
      if(cacheValid) {
        return cachedG[t > 0 ? y_+1 : 0][y][ex != null ? ex.x[t]+1 : 0];
      } else {
        return G_(t,y_,y,ex);
      }
    }
    double G__(int t, int y_, int y, Example ex) {
      double value = 0.;
      if( ex != null ) {
        value = expWeight[o(y, ex.x[t])];
        value *=  (t > 0) ? expWeight[t(y_, y)] : 1.;
        return value;
      } else {
        for(int x = 0; x < D; x++) {
          value *= expWeight[o(y, x)];
        }
        value *= (t > 0) ? expWeight[t(y_, y)] : 1.;
        return value;
      }
    }
    double G_(int t, int y_, int y, Example ex) {
      double value = 0.;
      if( ex != null ) {
        value = weights[o(y, ex.x[t])];
        value +=  (t > 0) ? weights[t(y_, y)] : 0.;
        return Math.exp(value);
      } else {
        for(int x = 0; x < D; x++) {
          double value_ = weights[o(y, x)];
          value += Math.exp(value_);
        }
        value *= (t > 0) ? Math.exp(weights[t(y_, y)]) : 1.;
        return value;
      }
    }

    public int o(int h, int x) {
      return h * D + x;
    }
    // Last K*K features are transitions
    public int t(int h_, int h) {
      return K * D + h_ * K + h;
    }

    @Override
    public int numGroups() {
      return K;
    }
  }

  public static Feature oFeature(int h, int x) {
    return new UnaryFeature(h, "x="+x);
  }
  public static Feature tFeature(int h_, int h) {
    return new BinaryFeature(h_, h);
  }

  final Indexer<Feature> featureIndexer;
  public UndirectedHiddenMarkovModel(int K, int D, int L) {
    this.K = K;
    this.D = D;
    this.L = L;


    // Careful - this must follow the same ordering as the index numbers
    this.featureIndexer = new Indexer<>();
    for(int h = 0; h < K; h++) {
      for(int x = 0; x < D; x++) {
        featureIndexer.add(oFeature(h, x));
        assert featureIndexer.indexOf(oFeature(h, x)) == o(h, x);
      }
    }
    for(int h_ = 0; h_ < K; h_++) {
      for(int h = 0; h < K; h++) {
        featureIndexer.add(tFeature(h_, h));
        assert featureIndexer.indexOf( tFeature(h_, h) )  == t(h_,h);
      }
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

  @Override
  public int numFeatures() {
    return K*D + K*K;
  }

  @Override
  public Parameters newParams() {
    return new Parameters(K, D, featureIndexer);
  }


  /**
   * Forward_t(y_{t}) = \sum_{y_{t-1}} G_t(y_{t-1}, y_t; x, \theta) Forward{t-1}(y_{t-1}).
   * TODO: No matter of threadsafe.
   */
  protected class IntermediateState {
    boolean use = false;
    Pair<Double, double[][]> forward;
    double[][] backward;

    void start() {
      use = true;
      forward = null;
      backward = null;
    }
    void stop() {
      use = false;
      forward = null;
      backward = null;
    }

  }
  protected final IntermediateState intermediateState = new IntermediateState();

  public Pair<Double, double[][]> forward(Parameters params, Example ex) {
    if(intermediateState.use && intermediateState.forward != null) return intermediateState.forward;
    int T = (ex != null) ? ex.x.length : L;
    double[][] forwards = new double[T][K];
    double A = 0; // Collected constants

    // Forward_0[k] = theta[BinaryFeature(-1, 0)];
    {
      // TODO: Support arbitrary initial features`
      for( int y = 0; y < K; y++ )
        forwards[0][y] = params.G(0, -1, y, ex);
      // Normalize
      double z = MatrixOps.sum(forwards[0]);
      MatrixOps.scale(forwards[0],1./z);
      A += Math.log(z);
    }

    // Forward_t[k] = \sum y_{t-1} Forward_[t-1][y_t-1] G(y_{t-1},y_t)
    for(int t = 1; t < T; t++) {
      for(int y = 0; y < K; y++){
        for(int y_ = 0; y_ < K; y_++) {
          forwards[t][y] += forwards[t-1][y_] * params.G(t, y_, y, ex);
        }
      }
      // Normalize
      double z = MatrixOps.sum(forwards[t]);
      MatrixOps.scale(forwards[t],1./z);
      A += Math.log(z);
    }
    return intermediateState.forward = Pair.newPair(A, forwards);
  }

  public double[][] backward(Parameters params, Example ex) {
    if(intermediateState.use && intermediateState.backward != null) return intermediateState.backward;

    int T = (ex != null) ? ex.x.length : L;
    double[][] backwards = new double[T][K];

    // Backward_{T-1}[k] = 1.
    {
      // TODO: Support arbitrary initial features`
      for( int y = 0; y < K; y++ )
        backwards[T-1][y] = 1.;
      // Normalize
      double z = MatrixOps.sum(backwards[T-1]);
      MatrixOps.scale(backwards[T-1],1./z);
    }

    // Backward_{T-1}[k] = \sum y_{t} Backward_[t][y_t] G(y_{t-1},y_t)
    for(int t = T-2; t >= 0; t--) {
      for(int y_ = 0; y_ < K; y_++) {
          for(int y = 0; y < K; y++){
          backwards[t][y_] += backwards[t+1][y] * params.G(t+1, y_, y, ex);
        }
      }
      // Normalize
      double z = MatrixOps.sum(backwards[t]);
      MatrixOps.scale(backwards[t],1./z);
    }

    return intermediateState.backward = backwards;
  }

  /**
   * Return p(y_{t-1}, y_t) 0 \le t \le T-1. The base case of p(y_{-1}, y_0) is when
   * y_{-1} is the -BEGIN- tag; in other words the initial probability of y_{-1}.
   */
  public double[][][] computeEdgeMarginals(Parameters params, Example ex) {
    int T = (ex != null) ? ex.x.length : L;

    double[][] forwards = forward(params,ex).getSecond();
    double[][] backwards = backward(params,ex);

    double[][][] marginals = new double[T][K][K];

    // T_0
    {
      int t = 0; int y_ = 0;
      for( int y = 0; y < K; y++ ) {
        marginals[t][y_][y] = params.G(t, y_, y, ex) * backwards[t][y];
      }
      // Normalize
      double z = MatrixOps.sum(marginals[t]);
      MatrixOps.scale(marginals[t], 1./z);
    }
    for( int t = 1; t < T; t++ ) {
      for(int y_ = 0; y_ < K; y_++){
        for(int y = 0; y < K; y++){
          marginals[t][y_][y] = forwards[t-1][y_] * params.G(t, y_, y, ex) * backwards[t][y];
        }
      }
      double z = MatrixOps.sum(marginals[t]);
      MatrixOps.scale(marginals[t], 1./z);
    }

    return marginals;
  }
  public double[][] computeHiddenNodeMarginals(Parameters params, Example ex) {
    int T = (ex != null) ? ex.x.length : L;

    double[][] forwards = forward(params,ex).getSecond();
    double[][] backwards = backward(params,ex);

    double[][] marginals = new double[T][K];

    for( int t = 0; t < T; t++) {
      for( int y = 0; y < K; y++) {
        marginals[t][y] += forwards[t][y] * backwards[t][y];
      }
      // Normalize
      double z = MatrixOps.sum(marginals[t]);
      MatrixOps.scale(marginals[t], 1./z);
    }

    return marginals;
  }
  public double[][][] computeEmissionMarginals(Parameters params, Example ex) {
    int T = (ex != null) ? ex.x.length : L;

    double[][] nodes = computeHiddenNodeMarginals(params, ex);

    double[][][] marginals;
    if(ex != null) {
      marginals = new double[T][K][1];
    } else {
      marginals = new double[T][K][D];
    }

    for( int t = 0; t < T; t++) {
      for( int y = 0; y < K; y++) {
        // Compute p(x|y)
        if(ex != null) {
          // Default update here
          marginals[t][y][0] = 1.;
          marginals[t][y][0] *= nodes[t][y]; // p(h)
        } else {
          for( int x = 0; x < D; x++)
            marginals[t][y][x] =
                    params.cacheValid
                            ? params.expWeight[o(y,x)]
                            : Math.exp(params.weights[o(y, x)]); // p(x|h)
          double z = MatrixOps.sum(marginals[t][y]);
          MatrixOps.scale(marginals[t][y], 1./z);
          for( int x = 0; x < D; x++)
            marginals[t][y][x] *= nodes[t][y]; // p(h)
        }
      }
    }
    return marginals;
  }

  /**
   * Use the Viterbi dynamic programming algorithm to find the hidden states for oFeature.
   * @return
   */
  public int[] viterbi( final Parameters params, final Example ex ) {
    assert ex != null;
    int T = ex.x.length;
    // Store the dynamic programming array and back pointers
    double [][] V = new double[T][K];
    int [][] Ptr = new int[T][K];

    // Initialize with 0 and path length
    {
      int  t = 0; int y_ = -1;
      for( int y = 0; y < K; y++ ) {
        // P( o_0 , s_k )
        V[t][y] = params.G(t, y_, y, ex);
        Ptr[t][y] = -1; // Doesn't need to be defined.
      }
      MatrixOps.normalize( V[0] );
    }

    // The dynamic program to find the optimal path
    for( int t = 1; t < T; t++ ) {
      for( int y = 0; y < K; y++ ) {
        int bestY = 0; double bestV = Double.NEGATIVE_INFINITY;
        for( int y_ = 0; y_ < K; y_++ ) {
          double v = V[t-1][y_] * params.G(t, y_, y, ex);
          if(v > bestV) {
            bestV = v; bestY = y_;
          }
        }
        V[t][y] = bestV; Ptr[t][y] = bestY;
      }
      assert !Double.isNaN(MatrixOps.sum(V[t])) && !Double.isInfinite(MatrixOps.sum(V[t]));
      MatrixOps.normalize( V[t] );
    }

    int[] z = new int[T];
    // Choose the best last state and back track from there
    z[T-1] = MatrixOps.argmax(V[T-1]);
    if( z[T-1] == -1 ) {
      LogInfo.log(Fmt.D(V));
      LogInfo.log(Fmt.D(T));
    }
    assert( z[T-1] != -1 );
    for(int i = T-1; i >= 1; i-- )  {
      assert( z[i] != -1 );
      z[i-1] = Ptr[i][z[i]];
    }

    return z;
  }
  public int[] viterbi( final Params params, final Example ex ) {
    if(params instanceof Parameters)
      return viterbi((Parameters) params, ex);
    else
      throw new IllegalArgumentException();
  }



  @Override
  public double getLogLikelihood(Params params, Example example) {
    if(params instanceof Parameters)
      return forward((Parameters) params, example).getFirst();
    else
      throw new IllegalArgumentException();
  }
  public double getLogLikelihood(Params params, int L) {
    int temp = this.L;
    this.L = L;
    double lhood = getLogLikelihood(params, (Example) null);
    this.L = temp;
    return lhood;
  }

  public double getFullProbability(Params params, Example ex) {
    if(!(params instanceof Parameters))
      throw new IllegalArgumentException();
    Parameters parameters = (Parameters) params;

    assert( ex.h != null );
    double lhood = 0.0;
    for(int t = 0; t < ex.x.length; t++ ) {
      int y = ex.h[t]; int x = ex.x[t];
      lhood += parameters.weights[o(y, x)];
    }
    for(int t = 1; t < ex.x.length; t++ ) {
      int y_ = ex.h[t-1]; int y = ex.x[t];
      lhood += parameters.weights[t(y_, y)];
    }

    return Math.exp( lhood - getLogLikelihood(parameters, (Example)null) );
  }

  @Override
  public void updateMarginals(Params params, int L, double scale, double count, Params marginals_) {
    int temp = this.L;
    this.L = L;
    updateMarginals(params, (Example)null, scale, count, marginals_);
    this.L = temp;
  }

  @Override
  public void updateMarginals(Params params, Example ex, double scale, double count, Params marginals_) {
    if(!(params instanceof Parameters))
      throw new IllegalArgumentException();
    if(!(marginals_ instanceof Parameters))
      throw new IllegalArgumentException();
    intermediateState.start();
    Parameters parameters = (Parameters) params;
    Parameters marginals = (Parameters) marginals_;

    int T = (ex != null) ? ex.x.length : L;

    double[][][] emissionMarginals = computeEmissionMarginals(parameters, ex);
    double[][][] edgeMarginals = computeEdgeMarginals(parameters, ex);

    // This will be a sparse update.
    for(int t = 0; t < T; t++) {
      // Add the emission marginal
      for( int y = 0; y < K; y++ ) {
        if( ex != null ) {
          int x = ex.x[t];
          marginals.weights[o(y, x)] += scale * emissionMarginals[t][y][0];
        } else {
          for( int x = 0; x < D; x++ ) {
            marginals.weights[o(y, x)] += scale * emissionMarginals[t][y][x];
          }
        }
      }
    }
    for(int t = 1; t < T; t++) {
      // Add the emission marginal
      for( int y = 0; y < K; y++ ) {
        // Add the transition marginal
        for( int y_ = 0; y_ < K; y_++ ) {
          marginals.weights[t(y_, y)] +=  scale * edgeMarginals[t][y_][y];
        }
      }
    }
    intermediateState.stop();
  }
  /**
   * Draw samples in this procedure;
   *    draw h_0, x_0, h_1, x_1, ...;
   *      - Compute p(h_0) and draw
   *      - drawing from p(h_0) is easy.
   *      - drawing from p(h_1 | h_2) is easy too.
   * @param params
   * @param genRandom
   * @param N
   * @return
   */
  @Override
  public Counter<Example> drawSamples(Params params, Random genRandom, int N) {
    if(!(params instanceof Parameters))
      throw new IllegalArgumentException();
    Parameters parameters = (Parameters) params;
    int T = L;

    // Pre-processing for edge and emission marginals.
    // Compute the edge marginals. This will let you draw the hs.
    double[][][] edgeMarginals = computeEdgeMarginals(parameters, null);
    for( int t = 0; t < T; t++ ) {
      for( int y_= 0; y_ < K; y_++ ) {
        MatrixOps.normalize(edgeMarginals[t][y_]);
      }
    }

    double[][] emissions = new double[K][D];
    for( int y = 0; y < K; y++ ) {
      for( int x = 0; x < D; x++ ) {
        emissions[y][x] = Math.exp( parameters.weights[o(y,x)] );
      }
      MatrixOps.normalize(emissions[y]);
    }

    Counter<Example> examples = new Counter<>();
    for( int i = 0; i < N; i++) {
      examples.add( drawSample(parameters, genRandom, edgeMarginals, emissions ) );
    }

    return examples;
  }

  public Example drawSample(Parameters parameters, Random genRandom, double[][][] edgeMarginals, double[][] emissions) {
    int T = L;
    Example ex = new Example(T);
    // Draw each $h_t$
    {
      int t = 0; int y_ = 0;
      ex.h[t] = RandomFactory.multinomial(genRandom, edgeMarginals[t][y_]);
    }
    for(int t = 1; t < T; t++) {
      int y_ = ex.h[t-1];
      ex.h[t] = RandomFactory.multinomial(genRandom, edgeMarginals[t][y_]);
    }
    // Draw each x_t
    for(int t = 0; t < T; t++) {
      int y = ex.h[t];
      ex.x[t] = RandomFactory.multinomial(genRandom, emissions[y]);
    }

    return ex;
  }

  public Parameters getSampleMarginals(Counter<Example> examples) {
    Parameters marginals = newParams();
    for(Example ex : examples) {
      for(int t = 0; t < ex.x.length; t++) {
        int y = ex.h[t]; int x = ex.x[t];
        marginals.weights[o(y, x)] += examples.getFraction(ex);
        if( t > 0 ) {
          int y_ = ex.h[t-1];
          marginals.weights[t(y_, y)] += examples.getFraction(ex);
        }
      }
    }
    return marginals;
  }


  /**
   * Choose idx
   */
  void generateExamples(Example current, int idx, List<Example> examples) {
    if( idx == current.x.length ) {
      examples.add(new Example(current.x));
    } else {
      // Make a choice for this index
      current.x[idx] = 0;
      generateExamples(current, idx+1, examples);
      current.x[idx] = 1;
      generateExamples(current, idx+1, examples);
    }
  }
  List<Example> generateExamples(int L) {
    List<Example> examples = new ArrayList<>((int)Math.pow(2,L));
    Example ex = new Example(new int[L]);
    generateExamples(ex, 0, examples);
    return examples;
  }

  @Override
  public Counter<Example> getDistribution(Params params) {
    Counter<Example> examples = new Counter<>();
    examples.addAll(generateExamples(L));
    for(Example ex: examples) {
      examples.set( ex, getProbability(params, ex));
    }
    return examples;
  }

  @Override
  public double updateMoments(Example ex, double count, SimpleMatrix P12, SimpleMatrix P13, SimpleMatrix P32, FullTensor P123) {
    double updates = 0.0;
    for(int start = 0; start < ex.x.length - 2; start++ ) {
      int x1 = ex.x[(start)];
      int x3 = ex.x[(start+1)]; // The most stable one we want to actually recover
      int x2 = ex.x[(start+2)];
      P13.set( x1, x3, P13.get( x1, x3 ) + count);
      P12.set( x1, x2, P12.get( x1, x2 ) + count);
      P32.set( x3, x2, P32.get( x3, x2 ) + count);
      P123.set( x1, x2, x3, P123.get(x1, x2, x3) + count );
      updates += count;
    }

    return updates;
  }

  public int getSize(Example ex) {
    return ex.x.length;
  }

}
