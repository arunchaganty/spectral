package learning.models.loglinear;

import fig.basic.Pair;
import learning.linalg.FullTensor;
import learning.linalg.MatrixOps;
import learning.linalg.RandomFactory;
import learning.models.ExponentialFamilyModel;
import learning.models.Params;
import learning.utils.Counter;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.Arrays;
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
    final int K;
    final int D;
    final double[] weights;

    public Parameters(int K, int D) {
      this.K = K;
      this.D = D;
      this.weights = new double[K*D + K*K];
    }

    @Override
    public Params newParams() {
      return new Parameters(K,D);
    }

    @Override
    public void initRandom(Random random, double noise) {
      for (int j = 0; j < weights.length; j++)
        weights[j] = noise * (2 * random.nextDouble() - 1);
    }

    @Override
    public void copyOver(Params other_) {
      if(other_ instanceof Parameters) {
        Parameters other = (Parameters) other_;
        System.arraycopy(other.weights, 0, weights, 0, weights.length);
      } else {
        throw new IllegalArgumentException();
      }
    }
    @Override
    public Params merge(Params other_) {
      // Nothing the merge, just add
      return plus(other_);
    }

    @Override
    public double[] toArray() {
      return weights;
    }
    @Override
    public int size() {
      return weights.length;
    }

    @Override
    public void clear() {
      Arrays.fill(weights, 0.);
    }

    @Override
    public void plusEquals(double scale, Params other_) {
      if(other_ instanceof Parameters) {
        Parameters other = (Parameters) other_;
        for(int i = 0; i < weights.length; i++)
          weights[i] += scale * other.weights[i];
      } else {
        throw new IllegalArgumentException();
      }

    }

    @Override
    public void scaleEquals(double scale) {
      for(int i = 0; i < weights.length; i++)
        weights[i] *= scale;
    }

    @Override
    public double dot(Params other_) {
      if(other_ instanceof Parameters) {
        Parameters other = (Parameters) other_;

        double prod = 0.;

        for(int i = 0; i < weights.length; i++)
          prod += weights[i] * other.weights[i];

        return prod;
      } else {
        throw new IllegalArgumentException();
      }
    }
  }

  public UndirectedHiddenMarkovModel(int K, int D, int L) {
    this.K = K;
    this.D = D;
    this.L = L;
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
  public Params newParams() {
    return new Parameters(K, D);
  }

  /**
   * Return \theta^T [\phi(y_{t-1}, y_{t}), \phi(y_{t}, x_{t}) ]
   */
  double G(Parameters params, int t, int y_, int y, Example ex) {
    double value = 0.;
    if( ex != null ) {
      value = params.weights[o(y, ex.x[t])];
      value +=  (t > 0) ? params.weights[t(y_, y)] : 0.;
      return Math.exp(value);
    } else {
      for(int x = 0; x < D; x++) {
        double value_ = params.weights[o(y, x)];
        value_ += (t > 0) ? params.weights[t(y_, y)] : 0.;
        value += Math.exp(value_);
      }
      return value;
    }
  }
  double logG(Parameters params, int t, int y_, int y, Example ex) {
    double value = Double.NEGATIVE_INFINITY;
    if( ex != null ) {
      value = params.weights[o(y, ex.x[t])];
      value +=  (t > 0) ? params.weights[t(y_, y)] : 0.;
      return value;
    } else {
      for(int x = 0; x < D; x++) {
        double value_ = params.weights[o(y, x)];
        value_ += (t > 0) ? params.weights[t(y_, y)] : 0.;
        value = MatrixOps.logsumexp(value, value_);
      }
      return value;
    }
  }

  /**
   * Forward_t(y_{t}) = \sum_{y_{t-1}} G_t(y_{t-1}, y_t; x, \theta) Forward{t-1}(y_{t-1}).
   * TODO: No matter of threadsafe.
   */
  private class IntermediateState {
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
  private final IntermediateState intermediateState = new IntermediateState();

//  public Pair<Double, double[][]> forward(Parameters params, Example ex) {
//    if(intermediateState.use && intermediateState.forward != null) return intermediateState.forward;
//    int T = (ex != null) ? ex.x.length : L;
//    double[][] forwards = new double[T][K];
//    double A = 0; // Collected constants
//
//    // Forward_0[k] = theta[BinaryFeature(-1, 0)];
//    {
//      // TODO: Support arbitrary initial features`
//      for( int y = 0; y < K; y++ )
//        forwards[0][y] = G(params, 0, -1, y, ex);
//      // Normalize
//      double z = MatrixOps.sum(forwards[0]);
//      MatrixOps.scale(forwards[0],1./z);
//      A += Math.log(z);
//    }
//
//    // Forward_t[k] = \sum y_{t-1} Forward_[t-1][y_t-1] G(y_{t-1},y_t)
//    for(int t = 1; t < T; t++) {
//      for(int y = 0; y < K; y++){
//        for(int y_ = 0; y_ < K; y_++) {
//          forwards[t][y] += forwards[t-1][y_] * G(params, t, y_, y, ex);
//        }
//      }
//      // Normalize
//      double z = MatrixOps.sum(forwards[t]);
//      MatrixOps.scale(forwards[t],1./z);
//      A += Math.log(z);
//    }
//    return intermediateState.forward = Pair.newPair(A, forwards);
//  }
  public Pair<Double, double[][]> forward(Parameters params, Example ex) {
//    if(intermediateState.use && intermediateState.forward != null) return intermediateState.forward;
    int T = (ex != null) ? ex.x.length : L;
    double[][] forwards = new double[T][K];
    for(int t = 0; t < T; t++)
      Arrays.fill(forwards[t], Double.NEGATIVE_INFINITY); // Zeros!
    double A = 0; // Collected constants

    // Forward_0[k] = theta[BinaryFeature(-1, 0)];
    {
      // TODO: Support arbitrary initial features`
      for( int y = 0; y < K; y++ )
        forwards[0][y] = logG(params, 0, -1, y, ex);
      // Normalize
      double z = MatrixOps.logsumexp(forwards[0]);
      MatrixOps.minus(forwards[0],z);
      A += z;
    }

    // Forward_t[k] = \sum y_{t-1} Forward_[t-1][y_t-1] G(y_{t-1},y_t)
    for(int t = 1; t < T; t++) {
      for(int y = 0; y < K; y++){
        for(int y_ = 0; y_ < K; y_++) {
          forwards[t][y] = MatrixOps.logsumexp(forwards[t][y], forwards[t-1][y_] + logG(params, t, y_, y, ex));
        }
      }
      // Normalize
      double z = MatrixOps.logsumexp(forwards[t]);
      MatrixOps.minus(forwards[t],z);
      A += z;
    }
    return intermediateState.forward = Pair.newPair(A, forwards);
  }

//  public double[][] backward(Parameters params, Example ex) {
//    if(intermediateState.use && intermediateState.backward != null) return intermediateState.backward;
//
//    int T = (ex != null) ? ex.x.length : L;
//    double[][] backwards = new double[T][K];
//
//    // Backward_{T-1}[k] = 1.
//    {
//      // TODO: Support arbitrary initial features`
//      for( int y = 0; y < K; y++ )
//        backwards[T-1][y] = 1.;
//      // Normalize
//      double z = MatrixOps.sum(backwards[T-1]);
//      MatrixOps.scale(backwards[T-1],1./z);
//    }
//
//    // Backward_{T-1}[k] = \sum y_{t} Backward_[t][y_t] G(y_{t-1},y_t)
//    for(int t = T-2; t >= 0; t--) {
//      for(int y_ = 0; y_ < K; y_++) {
//          for(int y = 0; y < K; y++){
//          backwards[t][y_] += backwards[t+1][y] * G(params, t+1, y_, y, ex);
//        }
//      }
//      // Normalize
//      double z = MatrixOps.sum(backwards[t]);
//      MatrixOps.scale(backwards[t],1./z);
//    }
//
//    return intermediateState.backward = backwards;
//  }
  public double[][] backward(Parameters params, Example ex) {
//    if(intermediateState.use && intermediateState.backward != null) return intermediateState.backward;

    int T = (ex != null) ? ex.x.length : L;
    double[][] backwards = new double[T][K];
    for(int t = 0; t < T; t++)
      Arrays.fill(backwards[t], Double.NEGATIVE_INFINITY); // Zeros!

    // Backward_{T-1}[k] = 1.
    {
      // TODO: Support arbitrary initial features`
      for( int y = 0; y < K; y++ )
        backwards[T-1][y] = 0.;
      // Normalize
      double z = MatrixOps.logsumexp(backwards[T-1]);
      MatrixOps.minus(backwards[T-1],z);
    }

    // Backward_{T-1}[k] = \sum y_{t} Backward_[t][y_t] G(y_{t-1},y_t)
    for(int t = T-2; t >= 0; t--) {
      for(int y_ = 0; y_ < K; y_++) {
        for(int y = 0; y < K; y++){
          backwards[t][y_] = MatrixOps.logsumexp(backwards[t][y_], backwards[t+1][y] + logG(params, t+1, y_, y, ex));
        }
      }
      // Normalize
      double z = MatrixOps.logsumexp(backwards[t]);
      MatrixOps.minus(backwards[t],z);
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
        marginals[t][y_][y] = logG(params, t, y_, y, ex) + backwards[t][y];
      }
      // Normalize
      double z = MatrixOps.logsumexp(marginals[t][y_]);
      for( int y = 0; y < K; y++ ) {
        marginals[t][y_][y] = Math.exp(marginals[t][y_][y] - z);
      }
    }
    for( int t = 1; t < T; t++ ) {
      for(int y_ = 0; y_ < K; y_++){
        for(int y = 0; y < K; y++){
          marginals[t][y_][y] = forwards[t-1][y_] + logG(params, t, y_, y, ex) + backwards[t][y];
        }
      }
      double z = Double.NEGATIVE_INFINITY;
      for(int y_ = 0; y_ < K; y_++){
        z = MatrixOps.logsumexp(z, MatrixOps.logsumexp(marginals[t][y_]));
      }
      for(int y_ = 0; y_ < K; y_++){
        for(int y = 0; y < K; y++){
          marginals[t][y_][y] = Math.exp(marginals[t][y_][y]  - z);
        }
      }
    }

    return marginals;
  }
  public double[][] computeHiddenNodeMarginals(Parameters params, Example ex) {
    int T = (ex != null) ? ex.x.length : L;

    double[][] forwards = forward(params,ex).getSecond();
    double[][] backwards = backward(params,ex);

    double[][] marginals = new double[T][K];
    for(int t = 0; t < T; t++)
      Arrays.fill(marginals[t], Double.NEGATIVE_INFINITY); // Zeros!

    for( int t = 0; t < T; t++) {
      for( int y = 0; y < K; y++) {
        marginals[t][y] = MatrixOps.logsumexp(marginals[t][y], forwards[t][y] + backwards[t][y]);
      }
      // Normalize
      double z = MatrixOps.logsumexp(marginals[t]);
      for( int y = 0; y < K; y++) {
        marginals[t][y] = Math.exp(marginals[t][y] - z);
      }
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
            marginals[t][y][x] = params.weights[o(y, x)]; // p(x|h)
          double z = MatrixOps.logsumexp(marginals[t][y]);
          // We want a distribution of p(x,y).
          for( int x = 0; x < D; x++)
            marginals[t][y][x] = Math.exp(marginals[t][y][x] - z) * nodes[t][y]; // p(h)
        }
      }
    }
    return marginals;
  }

  @Override
  public double getLogLikelihood(Params params, Example example) {
    if(params instanceof Parameters)
      return forward((Parameters) params, example).getFirst();
    else
      throw new IllegalArgumentException();
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
  public void updateMarginals(Params params, Example ex, double scale, Params marginals_) {
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
    Parameters marginals = new Parameters(K, D);
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

}
