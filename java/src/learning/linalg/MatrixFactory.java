/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu>
 * 
 */
package learning.linalg;

import org.ejml.simple.SimpleMatrix;

/**
 * Common routines to construct a number of different matrices
 */
public class MatrixFactory {

  /**
   * Convert from a double array to an integer one.
   * @param x
   * @return
   */
  public static double[] castToDouble( int[] x )
  {
    double[] x_ = new double[ x.length ];
    for( int i = 0; i < x.length; i++ ) x_[i] = x[i];
    return x_;
  }

  /**
   * Convert from an integer array to a double one.
   * @param x
   * @return
   */
  public static int[] castToInt( double[] x )
  {
    int[] x_ = new int[ x.length ];
    for( int i = 0; i < x.length; i++ ) x_[i] = (int) x[i];
    return x_;
  }

  /**
   * Create a simple matrix from a uni-dimensional array x
   * @param x
   * @return
   */
  public static SimpleMatrix fromVector( double[] x )
  {
    double[][] x_ = {x};
    return new SimpleMatrix( x_ );
  }

  /**
   * Create a simple matrix from a uni-dimensional array x
   * @param X
   * @return
   */
  public static double[] toVector( SimpleMatrix X )
  {
    if( X.numCols() == 1 )
      X = X.transpose();

    double[] M = new double[ X.numCols() ];
    for( int col = 0; col < X.numCols(); col++ ) {
      M[col] = X.get( 0, col );
    }

    return M;
  }

  /**
   * Create a simple matrix from a uni-dimensional array x
   * @param X
   * @return
   */
  public static double[][] toArray( SimpleMatrix X )
  {
    double[][] M = new double[ X.numRows() ][ X.numCols() ];
    for( int row = 0; row < X.numRows(); row++ ) {
      for( int col = 0; col < X.numCols(); col++ ) {
        M[row][col] = X.get( row, col );
      }
    }

    return M;
  }

  /**
   * Create a matrix of dimension n x m filled with zeros
   * @param n
   * @param m
   * @return
   */
  public static SimpleMatrix zeros( int n, int m ) {
    return new SimpleMatrix(n, m);
  }

  /**
   * Create a column vector of dimension n filled with zeros
   * @param n
   * @return
   */
  public static SimpleMatrix zeros( int n ) {
    return new SimpleMatrix(n, 1);
  }

  /**
   * Create a matrix of dimension n x m filled with ones
   * @param n
   * @param m
   * @return
   */
  public static SimpleMatrix ones( int n, int m ) {
    double[][] vals = new double[n][m];
    for( int i = 0; i < n; i++ )
      for( int j = 0; j < m; j++ )
        vals[i][j] = 1.0;
    return new SimpleMatrix(vals);
  }

  /**
   * Create a column vector of dimension n filled with ones
   * @param n
   * @return
   */
  public static SimpleMatrix ones( int n ) {
    double[][] vals = new double[n][1];
    for( int i = 0; i < n; i++ )
      vals[i][0] = 1.0;
    return new SimpleMatrix(vals);
  }

  /**
   * Create an identity matrix of dimension n x n
   * @param n
   * @return
   */
  public static SimpleMatrix eye( int n ) {
    return SimpleMatrix.identity(n);
  }

  /**
   * If M is a n x n matrix, return the diagonal elements
   * If M is a n x 1 matrix, return a matrix with the diagonal elements equal to M 
   * @param M
   * @return
   */
  public static SimpleMatrix diag( SimpleMatrix M ) {

    // If nx1 or 1xn, construct a 1x1 matrix
    if( M.numRows() == 1 || M.numCols() == 1 )
    {
      // Our standard is column vectors 
      if( M.numRows() == 1 ) M = M.transpose();

      int n = M.numRows();
      SimpleMatrix N = eye(n);
      for( int i = 0; i < n; i++ )
        N.set(i,i, M.get(i,0));
      return N;
    }
    else
    {
      // Extract the diagonal elements
      assert( M.numRows() == M.numCols() );

      int n = M.numRows();
      SimpleMatrix N =  zeros(n, 1);
      for( int i = 0; i < n; i++ )
        N.set(i,0, M.get(i,i));
      return N;
    }
  }	
  public static SimpleMatrix vectorStack( int n, SimpleMatrix M ) {
    assert( M.numRows() == 1 || M.numCols() == 1 );
    // Our standard is column vectors 
    if( M.numRows() == 1 ) M = M.transpose();

    int d = M.numRows();
    SimpleMatrix N = zeros(n, d);
    for( int i = 0; i < n; i++ )
      for( int j = 0; j < d; j++ )
        N.set(i, j, M.get(j,0));
    return N;
  }

  /**
   * Stack an array of vectors into a matrix column wise
   * @param M
   * @return
   */
  public static SimpleMatrix rowStack( SimpleMatrix[] M ) {
    int N = M.length;
    assert( M[0].numRows() == 1 || M[0].numCols() == 1 );
    int D = M[0].numRows() * M[0].numCols();

    double[][] X = new double[N][D];
    for( int n = 0; n < N; n++ ) {
      X[n] = MatrixFactory.toVector(M[n]);
    }
    return new SimpleMatrix(X);
  }
  public static SimpleMatrix columnStack( SimpleMatrix[] M ) {
    return rowStack(M).transpose();
  }

}

