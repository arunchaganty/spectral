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
public class UndirectedHiddenMarkovModelLogSpace extends UndirectedHiddenMarkovModel {
  public UndirectedHiddenMarkovModelLogSpace(int K, int D, int L) {
    super(K,D,L);
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
  public Pair<Double, double[][]> forward(Parameters params, Example ex) {
    if(intermediateState.use && intermediateState.forward != null) return intermediateState.forward;
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

  public double[][] backward(Parameters params, Example ex) {
    if(intermediateState.use && intermediateState.backward != null) return intermediateState.backward;

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
}
