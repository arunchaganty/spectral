package learning.models;

import fig.basic.Fmt;
import fig.basic.Indexer;
import fig.basic.LogInfo;
import fig.basic.Pair;
import learning.common.Utils;
import learning.linalg.FullTensor;
import learning.linalg.MatrixOps;
import learning.linalg.RandomFactory;
import learning.models.BasicParams;
import learning.models.ExponentialFamilyModel;
import learning.models.Params;
import learning.common.Counter;
import learning.models.loglinear.BinaryFeature;
import learning.models.loglinear.Example;
import learning.models.loglinear.Feature;
import learning.models.loglinear.UnaryFeature;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Undirected hidden Markov model
 */
public class HiddenMarkovModel extends ExponentialFamilyModel<Example> {
  public int K;
  public int D;
  public int L;

  // TODO: Support initial state
  public int pi(int h) {
    return h;
  }
  // Middle K*K features are transitions
  public int t(int h1, int h2) {
    return K + h1 * K + h2;
  }
  // Last K*D features are observations
  public int o(int h, int x) {
    return K + K * K + h * D + x;
  }

  public class Parameters extends BasicParams {
    public Parameters(int K, Indexer<Feature> featureIndexer) {
      super(K, featureIndexer);
    }

    @Override
    public void initRandom(Random random, double noise) {
      super.initRandom(random, noise);
      // Now normalize appropriately

      // - pi
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

      // - O
      for(int h1 = 0; h1 < K; h1++) {
        double z = 0.;
        for(int x = 0; x < D; x++) z += Math.abs(weights[o(h1,x)]);
        for(int x = 0; x < D; x++) weights[o(h1, x)] = Math.abs(weights[o(h1, x)])/z;
      }
    }

    public boolean isValid() {
      // - pi
      {
        double z = 0.;
        for(int h = 0; h < K; h++) z += Math.abs(weights[pi(h)]);
        if(!MatrixOps.equal(z,1.0)) return false;
      }

      // - T
      for(int h1 = 0; h1 < K; h1++) {
        double z = 0.;
        for(int h2 = 0; h2 < K; h2++) z += Math.abs(weights[t(h1,h2)]);
        if(!MatrixOps.equal(z,1.0)) return false;
      }

      // - O
      for(int h1 = 0; h1 < K; h1++) {
        double z = 0.;
        for(int x = 0; x < D; x++) z += Math.abs(weights[o(h1,x)]);
        if(!MatrixOps.equal(z,1.0)) return false;
      }

      return true;
    }

    public double G(int t, int y_, int y, Example ex) {
      double value = 0.;
      if( ex != null ) {
        value = weights[o(y, ex.x[t])];
        value *=  (t > 0) ? weights[t(y_, y)] : weights[pi(y)];
        return value;
      } else {
        // No contribution from O.
        value = (t > 0) ? weights[t(y_, y)] : weights[pi(y)];
        return value;
      }
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
    public double[][] getO() {
      double[][] O = new double[K][D];
      for(int h1 = 0; h1 < K; h1++)
        for(int x = 0; x < D; x++)
          O[h1][x] = weights[o(h1, x)];
      return O;
    }

  }

  public static Feature piFeature(int h) {
    return new UnaryFeature(h, "pi");
  }
  public static Feature oFeature(int h, int x) {
    return new UnaryFeature(h, "x="+x);
  }
  public static Feature tFeature(int h_, int h) {
    return new BinaryFeature(h_, h);
  }

  final Indexer<Feature> featureIndexer;
  public HiddenMarkovModel(int K, int D, int L) {
    this.K = K;
    this.D = D;
    this.L = L;


    // Careful - this must follow the same ordering as the index numbers
    this.featureIndexer = new Indexer<>();
    for(int h = 0; h < K; h++) {
      featureIndexer.add(piFeature(h));
      assert featureIndexer.indexOf( piFeature(h) )  == pi(h);
    }
    for(int h1 = 0; h1 < K; h1++) {
      for(int h2 = 0; h2 < K; h2++) {
        featureIndexer.add(tFeature(h1, h2));
        assert featureIndexer.indexOf( tFeature(h1, h2) )  == t(h1,h2);
      }
    }
    for(int h = 0; h < K; h++) {
      for(int x = 0; x < D; x++) {
        featureIndexer.add(oFeature(h, x));
        assert featureIndexer.indexOf(oFeature(h, x)) == o(h, x);
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
    return K + K*K + K*D;
  }

  @Override
  public Parameters newParams() {
    return new Parameters(K, featureIndexer);
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
      int t = 0;
      for( int y = 0; y < K; y++ )
        forwards[t][y] = params.G(t, -1, y, ex);
      // Normalize
      double z = MatrixOps.sum(forwards[t]);
      MatrixOps.scale(forwards[t],1./z);
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
      int t = T-1;
      for( int y = 0; y < K; y++ )
        backwards[t][y] = 1.;
      // Normalize
      double z = MatrixOps.sum(backwards[t]);
      MatrixOps.scale(backwards[t],1./z);
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

    for( int t = 0; t < T-1; t++ ) {
      for(int y_ = 0; y_ < K; y_++){
        for(int y = 0; y < K; y++){
          marginals[t][y_][y] = forwards[t][y_] * params.G(t+1, y_, y, ex) * backwards[t+1][y];
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
          marginals[t][y][0] = nodes[t][y]; // p(h)
        } else {
          for( int x = 0; x < D; x++)
            marginals[t][y][x] = params.weights[o(y, x)] * nodes[t][y]; // p(x|h) * p(h)
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

  @Override
  public double getLogLikelihood(Params params, Example example) {
    if(params instanceof Parameters)
      return forward((Parameters) params, example).getFirst();
    else
      throw new IllegalArgumentException();
  }
  public double getLogLikelihood(Params params, int L) {
    return 0; // Don't you love directed models?
  }

  public double getFullProbability(Params params_, Example ex) {
    if(!(params_ instanceof Parameters))
      throw new IllegalArgumentException();
    Parameters params = (Parameters) params_;

    assert( ex.h != null );
    double prob = 1.0;
    int y_ = -1;
    for(int t = 0; t < ex.x.length; t++ ) {
      int y = ex.h[t]; int x = ex.x[t];
      prob *= params.G(t, y_, y, ex);
      y_ = y;
    }

    return prob;
  }

  @Override
  public void updateMarginals(Params params, int L, double scale, Params marginals_) {
    int temp = this.L;
    this.L = L;
    updateMarginals(params, (Example)null, scale, marginals_);
    this.L = temp;
  }

  @Override
  public void updateMarginals(Params params, Example ex, double scale, Params marginals_) {
    if(!(params instanceof Parameters))
      throw new IllegalArgumentException();
    if(!(marginals_ instanceof Parameters))
      throw new IllegalArgumentException();
    intermediateState.start();
    Parameters parameters = (Parameters) params;
    Parameters marginals = (Parameters) marginals_;

    int T = (ex != null) ? ex.x.length : L;

    double[][] nodeMarginals = computeHiddenNodeMarginals(parameters, ex);
    double[][][] emissionMarginals = computeEmissionMarginals(parameters, ex);
    double[][][] edgeMarginals = computeEdgeMarginals(parameters, ex);

    // This will be a sparse update.
      // Add the emission marginal
    for( int y = 0; y < K; y++ ) {
      double z = 0;
      double updates = 0.;
      for(int t = 0; t < T; t++)
        z += nodeMarginals[t][y];
      for(int t = 0; t < T; t++) {
        if( ex != null ) {
          int x = ex.x[t];
          updates += nodeMarginals[t][y] / z;
          marginals.weights[o(y, x)] += scale * nodeMarginals[t][y] / z;
        } else {
          for( int x = 0; x < D; x++ ) {
            updates += emissionMarginals[t][y][x] / z;
            marginals.weights[o(y, x)] += scale * emissionMarginals[t][y][x] / z;
          }
        }
      }
      assert MatrixOps.equal(updates, 1.0);
    }
    {
      int t = 0;
      // Add the pi
      for( int y = 0; y < K; y++ ) {
        // Add the transition marginal
        marginals.weights[pi(y)] +=  scale * nodeMarginals[t][y];
      }
    }

    for( int y1 = 0; y1 < K; y1++ ) {
      for( int y2 = 0; y2 < K; y2++ ) {
        double z = 0.;
        for(int t = 0; t < T-1; t++) {
          z += nodeMarginals[t][y1];
        }
        for(int t = 0; t < T-1; t++) {
          marginals.weights[t(y1, y2)] +=  scale * edgeMarginals[t][y1][y2] / z;
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
   * @param params_
   * @param genRandom
   * @param N
   * @return
   */
  @Override
  public Counter<Example> drawSamples(Params params_, Random genRandom, int N) {
    if(!(params_ instanceof Parameters))
      throw new IllegalArgumentException();
    Parameters params = (Parameters) params_;

    double[] pi = params.getPi();
    double[][] T = params.getT();
    double[][] O = params.getO();

    Counter<Example> examples = new Counter<>();
    for( int i = 0; i < N; i++) {
      examples.add( drawSample(params, genRandom, pi, T, O ) );
    }

    return examples;
  }

  public Example drawSample(Params params, Random genRandom, double[] pi, double[][] T, double[][] O) {
    Example ex = new Example(L);
    // Draw each $h_t$
    ex.h[0] = RandomFactory.multinomial(genRandom, pi);
    for(int i = 1; i < L; i++) {
      int y_ = ex.h[i-1];
      ex.h[i] = RandomFactory.multinomial(genRandom, T[y_]);
    }
    // Draw each x_t
    for(int i = 0; i < L; i++) {
      int y = ex.h[i];
      ex.x[i] = RandomFactory.multinomial(genRandom, O[y]);
    }

    return ex;
  }

  public Parameters getSampleMarginals(Counter<Example> examples) {
    Parameters marginals = newParams();
    for(Example ex : examples) {
      for(int t = 0; t < ex.x.length; t++) {
        int y = ex.h[t]; int x = ex.x[t];
        marginals.weights[o(y, x)] += examples.getFraction(ex);
        if( t == 0 ) {
          marginals.weights[pi(y)] += examples.getFraction(ex);
        } else {
          int y_ = ex.h[t-1];
          marginals.weights[t(y_, y)] += examples.getFraction(ex);
        }
      }
    }

    // Now normalize...
    {
      double z = 0.;
      for(int y = 0; y < K; y++) z += marginals.weights[pi(y)];
      for(int y = 0; y < K; y++) marginals.weights[pi(y)] /= z;
    }
    for(int y1 = 0; y1 < K; y1++) {
      {
        double z = 0.;
        for(int x = 0; x < D; x++) z += marginals.weights[o(y1,x)];
        for(int x = 0; x < D; x++) marginals.weights[o(y1,x)] /= z;
      }
      {
        double z = 0.;
        for(int y2 = 0; y2 < D; y2++) z += marginals.weights[t(y1,y2)];
        for(int y2 = 0; y2 < D; y2++) marginals.weights[t(y1,y2)] /= z;
      }
    }

    return marginals;
  }

  @Override
  public Counter<Example> getDistribution(Params params) {
    Counter<Example> examples = new Counter<>();
    for(int[] h : Utils.enumerate(K, L) )
      for(int[] x : Utils.enumerate(D, L) )
        examples.add(new Example(x,h));
    for(Example ex: examples) {
      examples.set( ex, getFullProbability(params, ex));
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


}
