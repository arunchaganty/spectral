package learning.linalg;

import breeze.optimize.Minimizer;
import fig.basic.*;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;

import java.util.Random;

/**
 * Implements Anandkumar/Ge/Hsu/Kakade/Telgarsky, 2012.
 * Method of moments for learning latent-variable models.
 */
public class TensorFactorization {
  static final double EPS_CLOSE = 1e-10;

  @Option(gloss="Random number generator for tensor method and random projections")
  Random rnd = new Random();


  /**
   * Perform a single eigen-decomposition step
   * @param T - Full tensor
   * @return - (eigenvalue, eigenvector).
   */
  protected Pair<Double,SimpleMatrix> eigendecomposeStep( FullTensor T, int attempts, int iters ) {
    int N = iters;
    int D = T.getDim(0);

    double maxEigenvalue = Double.NEGATIVE_INFINITY;
    SimpleMatrix maxEigenvector = null;

    for( int attempt = 0; attempt < attempts; attempt++ ) {
      // 1. Draw theta randomly from unit sphere
      SimpleMatrix theta = RandomFactory.rand(rnd, 1, D);
      theta.scale(1.0/MatrixOps.norm(theta));
      if( attempt % 10 == 0 )
        LogInfo.logs("Attempt %d/%d", attempt, attempts);

      // 2. Compute power iteration update
      for(int n = 0; n < N; n++ ) {
        SimpleMatrix theta_ = T.project2(1, 2, theta, theta);
        // Normalize
        theta_ = theta_.scale(1.0 / MatrixOps.norm(theta_));
        double err = MatrixOps.norm(theta_.minus(theta));
        theta = theta_;
        if( err < EPS_CLOSE ) break;
      }
      double eigenvalue = T.project3(theta, theta, theta);
      if(eigenvalue > maxEigenvalue) {
        maxEigenvalue = eigenvalue;
        maxEigenvector = theta;
      }
    }
    SimpleMatrix theta = maxEigenvector;
    // 3. Do N iterations with this max eigenvector
    for(int n = 0; n < N; n++ ) {
      SimpleMatrix theta_ = T.project2(1, 2, theta, theta);
      // Normalize
      theta_ = theta_.scale(1.0 / MatrixOps.norm(theta_));
      double err = MatrixOps.norm(theta_.minus(theta));
      theta = theta_;
      if( err < EPS_CLOSE ) break;
    }
    double eigenvalue = T.project3(theta, theta, theta);

    return Pair.with(eigenvalue, theta);
  }

  @OptionSet(name="lbfgs") public LBFGSMaximizer.Options lbfgs = new LBFGSMaximizer.Options();
  @OptionSet(name="backtrack") public BacktrackingLineSearch.Options backtrack = new BacktrackingLineSearch.Options();

  /**
   * f(x) = \| T(x,x,x) - x^{3} \|.
   */
  public static class TensorFunctionState implements Maximizer.FunctionState {
    double[] point;
    FullTensor T;
    double value;
    double[] gradient;
    boolean objectiveValid = false;
    boolean gradientValid = false;

    TensorFunctionState(FullTensor T) {
      this.T = T;
      assert( T.D1 == T.D2 && T.D2 == T.D3 );
      point = new double[T.D1];
      gradient = new double[T.D1];
    }

    @Override
    public double[] point() {
      return point;
    }

    @Override
    public double value() {
      if( !objectiveValid ) {
        value = -(Math.pow(MatrixOps.norm(T),2.0) + Math.pow(MatrixOps.norm(point),6.0) - 2 * T.project3(point,point,point));
        objectiveValid = true;
      }
      return value;
    }

    @Override
    public double[] gradient() {
      if( !gradientValid ) {
        for( int i = 0; i < T.D1; i++ ) {
          gradient[i] = -(- 6 * T.project3(point, point, MatrixFactory.unitVector(T.D1, i)) +
              6 * Math.pow(MatrixOps.norm(point),4.0) * point[i]);
        }
        gradientValid = true;
      }
      return gradient;
    }

    @Override
    public void invalidate() {
      objectiveValid = gradientValid = false;
    }
  }

  protected Pair<Double,SimpleMatrix> eigendecomposeStep_( FullTensor T, int attempts, int iters ) {
    Maximizer max = new GradientMaximizer(backtrack); //LBFGSMaximizer(backtrack, lbfgs);
    TensorFunctionState state = new TensorFunctionState(T);
    Random rnd = new Random(1);
    for( int i = 0; i < state.point.length; i++ )
      state.point[i] = rnd.nextDouble();//maxEigenvector[i]; //1.0/(1+state.point.length);

    boolean done = false;
    LogInfo.begin_track("Optimization");
    for( int i = 0; i < iters && !done; i++ ) {
      done = max.takeStep(state);
      LogInfo.logs("%s objective = %f, point = %s, gradient = %s", i, state.value(), Fmt.D(state.point()), Fmt.D(state.gradient()));
    }
    LogInfo.end_track();
    SimpleMatrix eigenvector = MatrixFactory.fromVector(state.point());
    double eigenvalue = T.project3(eigenvector, eigenvector, eigenvector);

    return Pair.with(eigenvalue, eigenvector);
  }


  /**
   * Return T - scale vector^{\otimes 3}
   * "Efficient" inplace deflation
   * @param scale - remove $v$ with this scale
   * @param v - vector to remove
   * @return - Deflated tensor
   */
  protected void deflate(FullTensor T, double scale, SimpleMatrix vector) {
    int D = T.getDim(0);
    for( int d1 = 0; d1 < D; d1++ )
      for( int d2 = 0; d2 < D; d2++ )
        for( int d3 = 0; d3 < D; d3++ )
          T.set(d1,d2,d3, T.get(d1, d2, d3) - scale * vector.get(d1) * vector.get(d2) * vector.get(d3) );
  }

  /**
   * Find the first $k$ eigen-vectors and eigen-values of the tensor $T = \sum \lambda_i v_i^{\otimes 3}$
   * @param T - Tensor to be decomposed
   * @param K - largest K eigenvalues will be returned
   * @param attempts - number of times each eigendecomposition step is run (to pick the eigenvector of largest magnitude)
   * @param iters - Number of iterations to run the power method.
   * @return eigenvalues and eigenvectors
   */
  public Pair<SimpleMatrix, SimpleMatrix> eigendecompose( FullTensor T, int K, int attempts, int iters ) {
    int D = T.getDim(0);

    SimpleMatrix[] eigenvectors = new SimpleMatrix[K];
    double[] eigenvalues = new double[K];

    // Make a copy of T because we're going to destroy it during deflation
    FullTensor T_ = T.clone();

    for( int k = 0; k < K; k++ ) {
      // Extract the top eigenvalue/vector pair
      LogInfo.logs("Eigenvector %d/%d", k, K);
      Pair<Double, SimpleMatrix> pair = eigendecomposeStep(T_, attempts, iters);

      // When null, then we are done
      if( pair.getValue1() == null )
        break;

      eigenvalues[k] = pair.getValue0();
      eigenvectors[k] = pair.getValue1();

      // Deflate
      deflate(T_, eigenvalues[k], eigenvectors[k]);
    }

    SimpleMatrix eigenvalues_ = MatrixFactory.fromVector(eigenvalues);
    SimpleMatrix eigenvectors_ = MatrixFactory.columnStack(eigenvectors);

    // Make sure we have a factorization.
    {
      assert( K == eigenvalues_.getNumElements() );
      FullTensor Treconstructed = FullTensor.fromDecomposition( eigenvalues_, eigenvectors_ );
      LogInfo.logs( "T_: " + MatrixOps.diff(T, Treconstructed) );
    }

    return Pair.with(eigenvalues_, eigenvectors_);
  }
  public Pair<SimpleMatrix, SimpleMatrix> eigendecompose( FullTensor T, int K ) {
    return eigendecompose(T, K, 100, 50);
  }
  public Pair<SimpleMatrix, SimpleMatrix> eigendecompose( FullTensor T ) {
    int K = T.getDim(0);
    return eigendecompose(T, K);
  }

  /**
   * Check to see that v is indeed an eigenvector of T
   */
  protected boolean isEigenvector( Tensor T, SimpleMatrix v ) {
    // Firstly, normalize $v$ - to be sure.
    v = MatrixOps.normalize(v);
    SimpleMatrix u = T.project2( 1, 2, v, v );
    u = MatrixOps.normalize(u);
    return MatrixOps.allclose( u, v );
  }

}
