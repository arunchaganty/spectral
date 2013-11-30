package learning.models.loglinear;

import fig.basic.Indexer;
import fig.basic.Pair;
import learning.linalg.MatrixOps;
import learning.linalg.RandomFactory;
import learning.models.ExponentialFamilyModel;
import learning.utils.Counter;

import java.util.Random;

/**
 * Undirected hidden Markov model
 */
public class UndirectedHiddenMarkovModel implements ExponentialFamilyModel<Example> {
  public int K;
  public int D;
  public int L;
  Indexer<Feature> indexer;

  private static Feature o(int h, int x) {
    return new UnaryFeature(h, "x="+x);
  }
  private static Feature t(int h, int h_) {
    return new BinaryFeature(h, h_);
  }

  public UndirectedHiddenMarkovModel(int K, int D, int L) {
    this.K = K;
    this.D = D;
    this.L = L;

    for(int h = 0; h < K; h++) {
      for(int x = 0; x < D; x++) {
        indexer.add(o(h,x));
      }
      if( L > 1 ) {
        for(int h_ = 0; h_ < K; h_++) {
          indexer.add(t(h_,h));
        }
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
    return indexer.size();
  }

  @Override
  public ParamsVec newParamsVec() {
    return new ParamsVec(K, indexer);
  }

  // TODO: compute forward and backward messages (marginals)

  /**
   * Return \theta^T [\phi(y_{t-1}, y_{t}), \phi(y_{t}, x_{t}) ]
   * @param params
   * @param t
   * @param y_
   * @param y
   * @param ex
   * @return
   */
  double G(ParamsVec params, int t, int y_, int y, Example ex) {
    double value = 0.;
    if( ex != null ) {
      value = params.get(o(y,ex.x[t]));
      value *=  (t > 0) ? params.get(t(y_, y)) : 1.;
    } else {
      for(int x = 0; x < D; x++) value += params.get(o(y,x));
      value *= (t > 0) ? params.get(t(y_, y)) : 1.;
    }
    return value;
  }

  /**
   * Forward_t(y_{t}) = \sum_{y_{t-1}} G_t(y_{t-1}, y_t; x, \theta) Forward{t-1}(y_{t-1}).
   * @return
   */
  public Pair<Double, double[][]> forward(ParamsVec params, Example ex) {
    int T = (ex != null) ? ex.x.length : L;
    double[][] forwards = new double[T][K];
    double A = 0; // Collected constants

    // Forward_0[k] = theta[BinaryFeature(-1, 0)];
    {
      // TODO: Support arbitrary initial features`
      for( int y = 0; y < K; y++ )
        forwards[0][y] = G(params, 0, -1, y, ex);
      // Normalize
      double z = MatrixOps.sum(forwards[0]);
      MatrixOps.scale(forwards[0],1./z);
      A += Math.log(z);
    }

    // Forward_t[k] = \sum y_{t-1} Forward_[t-1][y_t-1] G(y_{t-1},y_t)
    for(int t = 1; t < T; t++) {
      for(int y = 0; y < K; y++){
        for(int y_ = 0; y_ < K; y_++) {
          forwards[t][y] += forwards[t-1][y_] * G(params, t, y_, y, ex);
        }
      }
      // Normalize
      double z = MatrixOps.sum(forwards[t]);
      MatrixOps.scale(forwards[t],1./z);
      A += Math.log(z);
    }

    return Pair.makePair(A, forwards);
  }

  public double[][] backward(ParamsVec params, Example ex) {
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
    for(int t = T-2; t <= 0; t--) {
      for(int y_ = 0; y_ < K; y_++) {
          for(int y = 0; y < K; y++){
          backwards[t][y_] += backwards[t+1][y] * G(params, t, y_, y, ex);
        }
      }
      // Normalize
      double z = MatrixOps.sum(backwards[t]);
      MatrixOps.scale(backwards[t],1./z);
    }

    return backwards;
  }

  /**
   * Return p(y_{t-1}, y_t) 0 \le t \le T-1. The base case of p(y_{-1}, y_0) is when
   * y_{-1} is the -BEGIN- tag; in other words the initial probability of y_{-1}.
   * @param params
   * @param ex
   * @return
   */
  public double[][][] computeEdgeMarginals(ParamsVec params, Example ex) {
    int T = (ex != null) ? ex.x.length : L;

    double[][] forwards = forward(params,ex).getSecond();
    double[][] backwards = backward(params,ex);

    double[][][] marginals = new double[T][K][K];

    // T_0
    {
      int t = 0;
      int y_ = 0;
      for( int y = 0; y < K; y++ ) {
        marginals[0][y_][y] = G(params, t, y_, y, ex) * backwards[0][y];
      }
      // Normalize
      double z = MatrixOps.sum(marginals[t]);
      MatrixOps.scale(marginals[t], 1./z);
    }
    for( int t = 1; t < T; t++ ) {
      for(int y_ = 0; y_ < K; y_++){
        for(int y = 0; y < K; y++){
          marginals[t][y_][y] = forwards[t-1][y_] * G(params, t, y_, y, ex) * backwards[t][y];
        }
      }
      double z = MatrixOps.sum(marginals[t]);
      MatrixOps.scale(marginals[t], 1./z);
    }

    return marginals;
  }
  public double[][] computeHiddenNodeMarginals(ParamsVec params, Example ex) {
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
  public double[][][] computeEmissionMarginals(ParamsVec params, Example ex) {
    int T = (ex != null) ? ex.x.length : L;

    double[][] nodes = computeHiddenNodeMarginals(params, ex);

    double[][][] marginals = new double[T][K][D];

    for( int t = 0; t < T; t++) {
      for( int y = 0; y < K; y++) {
        for( int x = 0; x < D; x++) {
          marginals[t][y][x] = nodes[t][y]; // p(h)
          if( ex != null ) {
            marginals[t][y][x] *= (x == ex.x[t]) ? 1. : 0.; // p(x|h)
          } else {
            marginals[t][y][x] *= params.get(o(y, x)); // p(x|h)
          }
        }
        // Normalize; we want a distribution of p(y,x).
        double z = MatrixOps.sum(marginals[t]);
        MatrixOps.scale(marginals[t], 1./z);
      }
    }
    return marginals;
  }

  @Override
  public double getLogLikelihood(ParamsVec parameters, Example example) {
    return forward(parameters, example).getFirst();
  }

  @Override
  public double getLogLikelihood(ParamsVec parameters) {
    return getLogLikelihood(parameters, (Example)null);
  }

  @Override
  public double getLogLikelihood(ParamsVec parameters, Counter<Example> examples) {
    double lhood = 0.;
    for(Example ex : examples) {
      lhood += examples.getCount(ex) / examples.sum() * getLogLikelihood(parameters, ex);
    }
    return lhood;
  }

  @Override
  public ParamsVec getMarginals(ParamsVec parameters, Example ex) {
    double[][][] emissionMarginals = computeEmissionMarginals(parameters, ex);
    double[][][] edgeMarginals = computeEdgeMarginals(parameters, ex);

    int T = (ex != null) ? ex.x.length : L;

    ParamsVec marginals = newParamsVec();
    for(int t = 0; t < T; t++) {
      // Add the emission marginal
      for( int y = 0; y < K; y++ ) {
        if( ex != null ) {
          marginals.set(o(y,ex.x[t]), emissionMarginals[t][y][ex.x[t]]);
        } else {
          for( int x = 0; x < D; x++ ) {
            marginals.set(o(y,x), emissionMarginals[t][y][x]);
          }
        }
        // Add the transition marginal
        if( t > 0 ) {
          for( int y_ = 0; y_ < K; y_++ ) {
            if( ex != null ) {
              marginals.set(t(y_,y), edgeMarginals[t][y_][y]);
            }
          }
        }
      }
    }

    return marginals;
  }

  @Override
  public ParamsVec getMarginals(ParamsVec parameters) {
    return getMarginals(parameters, (Example) null);
  }

  @Override
  public ParamsVec getMarginals(ParamsVec parameters, Counter<Example> examples) {
    ParamsVec marginals = newParamsVec();
    for(Example ex: examples) {
      ParamsVec marginals_ = getMarginals(parameters, ex);
      marginals.incr( examples.getCount(ex) / examples.sum(), marginals_ );
    }
    return marginals;  //To change body of implemented methods use File | Settings | File Templates.
  }

  /**
   * Draw samples in this procedure;
   *    draw h_0, x_0, h_1, x_1, ...;
   *      - Compute p(h_0) and draw
   *      - drawing from p(h_0) is easy.
   *      - drawing from p(h_1 | h_2) is easy too.
   * @param parameters
   * @param genRandom
   * @return
   */
  @Override
  public Counter<Example> drawSamples(ParamsVec parameters, Random genRandom, int N) {
    int T = L;

    // Pre-processing for edge and emission marginals.
    // Compute the edge marginals. This will let you draw the hs.
    double[][][] edgeMarginals = computeEdgeMarginals(parameters, null);
    for( int t = 0; t < T; t++ ) {
      for( int y = 0; y < K; y++ ) {
        MatrixOps.normalize(edgeMarginals[t][y]);
      }
    }

    double[][] emissions = new double[K][D];
    for( int y = 0; y < K; y++ ) {
      for( int x = 0; x < D; x++ ) {
        emissions[y][x] = parameters.get(o(y,x));
      }
      MatrixOps.normalize(emissions[y]);
    }

    Counter<Example> examples = new Counter<>();
    for( int i = 0; i < N; i++) {
      examples.add( drawSample(parameters, genRandom, edgeMarginals, emissions ) );
    }

    return examples;
  }

  public Example drawSample(ParamsVec parameters, Random genRandom, double[][][] edgeMarginals, double[][] emissions) {
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
    for(int t = 1; t < T; t++) {
      int y = ex.h[t];
      ex.x[t] = RandomFactory.multinomial(genRandom, emissions[y]);
    }

    return ex;
  }

  @Override
  public Example drawSample(ParamsVec parameters, Random genRandom) {
    return drawSamples(parameters, genRandom, 1).iterator().next();
  }

}
