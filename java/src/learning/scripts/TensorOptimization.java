package learning.scripts;

import fig.basic.*;
import fig.exec.Execution;
import learning.linalg.*;
import org.ejml.simple.SimpleMatrix;

import java.util.Random;

/**
 * Stupid script to optimize tensors
 */
public class TensorOptimization implements  Runnable {

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


  protected void factorize( FullTensor T, double[] maxEigenvector, int iters ) {
    Maximizer max = new GradientMaximizer(backtrack); //LBFGSMaximizer(backtrack, lbfgs);
    TensorFunctionState state = new TensorFunctionState(T);
    Random rnd = new Random(1);
    for( int i = 0; i < state.point.length; i++ )
      state.point[i] = rnd.nextDouble();//maxEigenvector[i]; //1.0/(1+state.point.length);

    boolean done = false;
    LogInfo.begin_track("Optimization");
    for( int i = 0; i < iters && !done; i++ ) {
      done = max.takeStep(state);
      // Project state back onto unit sphere.
//      MatrixOps.makeUnitVector(state.point);
//      state.invalidate();

      LogInfo.logs("%s objective = %f, point = %s, gradient = %s", i, state.value(), Fmt.D(state.point()), Fmt.D(state.gradient()));
//      LogInfo.logs("%s objective = %f", i, state.value());
    }
    LogInfo.end_track();
    SimpleMatrix eigenvector = MatrixFactory.fromVector(state.point());
    //eigenvector = MatrixOps.makeUnitVector(T.project2(1, 2, eigenvector, eigenvector));

    //FullTensor T_ = FullTensor.fromDecomposition( new double[]{1.0}, new double[][]{state.point()} );
    FullTensor T_ = FullTensor.fromUnitVector(eigenvector);
    // Print out $x$
    LogInfo.logs( Fmt.D(state.point()) );
    LogInfo.logs( MatrixOps.maxdiff(T, T_) );
    LogInfo.logs( MatrixOps.norm(T.minus(T_)) );

//    SimpleMatrix eigenvector_ = MatrixOps.makeUnitVector(T.project2(1, 2, eigenvector, eigenvector));
//    LogInfo.logs( eigenvector_ );
//    LogInfo.logs( MatrixOps.diff( eigenvector, eigenvector_ ) );

//    LogInfo.logs( T );
//    LogInfo.logs( T_ );
  }

  public void run() {
    // Stupid thing.
    FullTensor T;
    double[] x = new double[] {1.0, 0.0, 0.0};
//    T = FullTensor.fromDecomposition(
//        new double[]{1.0},
//        new double[][]{x});
//    factorize(T, x, 10);
//    // Less stupid thing.
//    T = FullTensor.fromDecomposition(
//        new double[]{0.5},
//        new double[][]{x});
//    factorize(T, x, 100);
//
    Random rnd = new Random(1);

    double[] y;
    x = RandomFactory.rand_(rnd, 3);
    y = RandomFactory.rand_(rnd, 3);
    MatrixOps.makeUnitVector(x);
    MatrixOps.makeUnitVector(y);

    T = FullTensor.fromDecomposition(
        new double[]{1.0},
        new double[][]{x});
    factorize(T, x, 100);
    LogInfo.logs(Fmt.D(x));

    T = FullTensor.fromDecomposition(
        new double[]{1.0, 1.0},
        new double[][]{x,y});
    factorize(T, x, 100);
    LogInfo.logs(Fmt.D(x));
    LogInfo.logs(Fmt.D(y));

    SimpleMatrix w = new SimpleMatrix( new double[][] {{ 1.0, 1.0, 1.0, 1.0 }} );
    SimpleMatrix X = RandomFactory.orthogonal(4).plus( RandomFactory.randn(4,4).scale(0.5) );
    T = FullTensor.fromDecomposition( w, X );
    LogInfo.logs(X);
    x = MatrixFactory.toVector(MatrixOps.col(X,0));
    y = MatrixFactory.toVector(MatrixOps.col(X,1));
    factorize(T, x, 100);
    LogInfo.logs( X ); //Fmt.D(x));
//    LogInfo.logs(Fmt.D(y));


  }

  public static void main(String[] args) {
    Execution.run(args, new TensorOptimization() );
  }


}
