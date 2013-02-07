/**
 * learning.linalg
 * Arun Chaganty (chaganty@stanford.edu)
 *
 */

package learning.linalg;

import learning.linalg.SimpleTensor;
import learning.exceptions.NumericalException;

import org.ejml.simple.SimpleMatrix;
import org.ejml.data.DenseMatrix64F;
import org.ejml.simple.SimpleSVD;
import org.ejml.simple.SimpleEVD;
import org.javatuples.*;

public class MatrixOps {

  public static double EPS_CLOSE = 1e-4;
  public static double EPS_ZERO = 1e-7;

  /**
   * Print entries of a vector
   */
  public static void printVector( double[] x ) {
    System.out.printf( "{ " );
    for( int i = 0; i < x.length; i++ )
      System.out.printf( "%f, ", x[i] );
    System.out.printf( "}\n" );
  }
  public static void printVector( int[] x ) {
    System.out.printf( "{ " );
    for( int i = 0; i < x.length; i++ )
      System.out.printf( "%d, ", x[i] );
    System.out.printf( "}\n" );
  }
  public static void printSize( SimpleMatrix X ) {
    System.out.printf( "(%d, %d)\n", X.numRows(), X.numCols() );
  }
  public static void printSize( Tensor X ) {
    System.out.printf( "(%d, %d, %d)\n", X.getDim(0), X.getDim(1), X.getDim(2) );
  }
  public static void printSize( double[][] X ) {
    System.out.printf( "(%d, %d)\n", X.length, X[0].length );
  }
  public static void printSize( double[] X ) {
    System.out.printf( "(%d,)\n", X.length );
  }

  /**
   * Print entries of a arrays
   */
  public static String arrayToString( double[][] X ) {
    String out = "";
    out += "{\n";
    for( int i = 0; i < X.length; i++ ) {
      out += "{ ";
      for( int j = 0; j < X[i].length; j++ )
        out += String.valueOf(X[i][j]) + ", ";
      out += "}\n";
    }
    out += "}";

    return out;
  }
  public static String arrayToString( int[][] X ) {
    String out = "";
    out += "{\n";
    for( int i = 0; i < X.length; i++ ) {
      out += "{ ";
      for( int j = 0; j < X[i].length; j++ )
        out += String.valueOf(X[i][j]) + ", ";
      out += "}\n";
    }
    out += "}";

    return out;
  }
  public static void printArray( double[][] X ) {
    System.out.println( arrayToString(X));
  }
  public static void printArray( int[][] X ) {
    System.out.println( arrayToString(X));
  }

  /**
   * Take the vector dot product of two vectors (as arrays)
   */
  public static double dot( double[] x, double[] y ) {
    assert( x.length == y.length );
    double sum = 0.0;
    for( int i = 0; i < x.length; i++ )
      sum += x[i] * y[i];
    return sum;
  }

  /**
   * Test whether two matrices are within eps of each other
   */
  public static boolean allclose( double[] X1, double[] X2, double eps ) {
    for( int i = 0; i < X1.length; i++ ) {
      if( Math.abs( X1[i] - X2[i] )  > eps ) {
        return false;
      }
    }
    return true;
  }
  public static boolean allclose( double[] X1, double[] X2 ) {
    return allclose( X1, X2, EPS_CLOSE );
  }

  public static boolean equal( int[] X1, int[] X2 ) {
    if( X1.length != X2.length ) return false;
    for( int i = 0; i < X1.length; i++ ) {
      if( X1[i] != X2[i] ) return false;
    }
    return true;
  }
  public static boolean equal( double x1, double x2, double eps ) {
    return Math.abs( x1 - x2  ) < eps;
  }
  public static boolean equal( double x1, double x2 ) {
    return equal( x1, x2, EPS_ZERO );
  }

  public static boolean isVector( DenseMatrix64F X ) {
    return X.numCols == 1 || X.numRows == 1;
  }
  public static boolean isVector( SimpleMatrix X ) {
    return X.numCols() == 1 || X.numRows() == 1;
  }

  /**
   * Test whether two matrices are within eps of each other
   */
  public static boolean allclose( DenseMatrix64F X1, DenseMatrix64F X2, double eps ) {
    if( isVector(X1) && isVector(X2) )
      return allclose( X1.data, X2.data, eps );
    assert( X1.numRows == X2.numRows );
    assert( X1.numCols == X2.numCols );

    double[] X1_ = X1.data;
    double[] X2_ = X2.data;

    for( int i = 0; i < X1_.length; i++ ) {
      if( Math.abs( X1_[i] - X2_[i] )  > eps ) {
        return false;
      }
    }

    return true;
  }
  public static boolean allclose( DenseMatrix64F X1, DenseMatrix64F X2 ) {
    return allclose( X1, X2, EPS_CLOSE );
  }
  public static boolean allclose( SimpleMatrix X1, SimpleMatrix X2, double eps ) {
    return allclose( X1.getMatrix(), X2.getMatrix(), eps );
  }
  public static boolean allclose( SimpleMatrix X1, SimpleMatrix X2 ) {
    return allclose( X1.getMatrix(), X2.getMatrix() );
  }


  /**
   * Find the norm of a matrix
   */
  public static double norm(DenseMatrix64F X) {
    double sum = 0.0;
    double[] X_ = X.data;
    return norm( X_ );
  }
  public static double norm(SimpleMatrix X) {
    return norm( X.getMatrix() );
  }

  /**
   * Return the absolute value of each entry in X
   */
  public static void abs( DenseMatrix64F X ) {
    double[] X_ = X.data;
    for( int i = 0; i < X_.length; i++ ) 
      X_[i] = Math.abs( X_[i] );
  }

  public static SimpleMatrix abs( SimpleMatrix X ) {
    DenseMatrix64F Y = X.getMatrix().copy();
    abs(Y);
    return SimpleMatrix.wrap( Y ) ;
  }

  /**
   * Find the minimium value of the matrix X
   */
  public static double min( double[] x ) {
    double min = Double.POSITIVE_INFINITY;
    for( int i = 0; i < x.length; i++ ) 
      if( x[i] < min ) min = x[i];

    return min;
  }
  public static double min( DenseMatrix64F X ) {
    return min( X.data );
  }
  public static double min( SimpleMatrix X ) {
    return min( X.getMatrix() );
  }
  /**
   * Find the maximum value of the matrix X
   */
  public static double max( double[] x ) {
    double max = Double.NEGATIVE_INFINITY;
    for( int i = 0; i < x.length; i++ ) 
      if( x[i] > max ) max = x[i];

    return max;
  }
  public static double max( DenseMatrix64F X ) {
    return max( X.data );
  }
  public static double max( SimpleMatrix X ) {
    return max( X.getMatrix() );
  }

  public static double maxAbs( double[] x ) {
    double max = 0.0;
    for( int i = 0; i < x.length; i++ )
      if( Math.abs(x[i]) > max ) max = Math.abs(x[i]);

    return max;
  }
  public static double minAbs( double[] x ) {
    double min = Double.POSITIVE_INFINITY;
    for( int i = 0; i < x.length; i++ )
      if( Math.abs(x[i]) < min ) min = Math.abs(x[i]);

    return min;
  }

  /**
   * Find the location of the minimum value of X
   */
  public static int argmax( double[] x ) {
    int idx = -1;
    double max = Double.NEGATIVE_INFINITY;
    for( int i = 0; i < x.length; i++ ) 
      if( x[i] > max ) {
        idx = i;
        max = x[i];
      }

    return idx;
  }
  public static int argmax( DenseMatrix64F X ) {
    return argmax( X.data );
  }
  public static int argmax( SimpleMatrix X ) {
    return argmax( X.getMatrix() );
  }
  public static int argmin( double[] x ) {
    int idx = -1;
    double min = Double.POSITIVE_INFINITY;
    for( int i = 0; i < x.length; i++ ) 
      if( x[i] < min ) {
        idx = i;
        min = x[i];
      }

    return idx;
  }
  public static int argmin( DenseMatrix64F X ) {
    return argmin( X.data );
  }
  public static int argmin( SimpleMatrix X ) {
    return argmin( X.getMatrix() );
  }



  /**
   * Compute the average outer product of each row of X1 and X2
   */
  public static DenseMatrix64F Pairs( DenseMatrix64F X1, DenseMatrix64F X2 ) {
    assert( X1.numRows == X2.numRows );

    int nRows = X1.numRows;
    int n = X1.numCols;
    int m = X2.numCols;

    double[] X1_ = X1.data;
    double[] X2_ = X2.data;
    double[][] Z = new double[n][m];

    // Average the outer products
    for(int row = 0; row < nRows; row++ ) {
      for( int i = 0; i < n; i++ ) {
        double x1 = X1_[X1.getIndex(row, i)];
        for( int j = 0; j < m; j++ ) {
          double x2 = X2_[X2.getIndex(row, j)];
          // Rolling average
          Z[i][j] += (x1*x2 - Z[i][j])/(row+1);
        }
      }
    }

    return new DenseMatrix64F( Z );
  }
  public static SimpleMatrix Pairs( SimpleMatrix X1, SimpleMatrix X2 ) {
    return SimpleMatrix.wrap( Pairs( X1.getMatrix(), X2.getMatrix() ) );
  }

  /**
   * Compute Triples
   */
  public static SimpleTensor Triples( SimpleMatrix X1, SimpleMatrix X2, SimpleMatrix X3 ) {
    return new SimpleTensor( X1, X2, X3 );
  }

  /**
   * Extract the i-th column of X
   */
  public static SimpleMatrix col( SimpleMatrix X, int col ) {
    return X.extractMatrix( 0, SimpleMatrix.END, col, col+1 );
  }
  public static DenseMatrix64F col( DenseMatrix64F X, int col ) {
    DenseMatrix64F x = new DenseMatrix64F( X.numRows, 1 );
    for( int row = 0; row < X.numRows; row++ )
      x.set(row, X.get( X.getIndex(row, col)));
    return x;
  }

  /**
   * Extract the i-th row of X
   */
  public static SimpleMatrix row( SimpleMatrix X, int row ) {
    return X.extractMatrix( row, row+1, 0, SimpleMatrix.END);
  }
  public static DenseMatrix64F row( DenseMatrix64F X, int row ) {
    DenseMatrix64F x = new DenseMatrix64F( 1, X.numCols );
    for( int col = 0; col < X.numCols; col++ )
      x.set(col, X.get( X.getIndex(row, col)));
    return x;
  }

  /**
   * Set the rows i to j of the matrix X
   * @param X
   * @param i
   * @param j
   * @return
   */
  public static void setRows( SimpleMatrix X, int i, int j, SimpleMatrix r ) {
    assert( r.numRows() == j-i );
    for( int row = i; row < j; row++ )
    {
      for( int col = 0; col < r.numCols(); col++ )
        X.set( row, col, r.get(row-i,col));
    }
  }
  public static void setRow( SimpleMatrix X, int i, SimpleMatrix r ) {
    if( r.numCols() == 1 )
      r = r.transpose();
    setRows( X, i, i+1, r );
  }
  public static void setRow( SimpleMatrix X, int i, double[] r ) {
    assert( X.numCols() == r.length );
    for( int col = 0; col < X.numCols(); col++ )
      X.set( i, col, r[col] );
  }

  /**
   * Set the cols i to j of the matrix X
   * @param X
   * @param i
   * @param j
   * @return
   */
  public static void setCols( SimpleMatrix X, int i, int j, SimpleMatrix c ) {
    assert( c.numCols() == j-i );
    for( int col = i; col < j; col++ )
    {
      for( int row = 0; row < c.numRows(); row++ )
        X.set( row, col, c.get(row,col-i));
    }
  }
  /**
   * Set the i-th column of the matrix X
   * @param X
   * @param i
   * @return
   */
  public static void setCol( SimpleMatrix X, int i, SimpleMatrix c ) {
    if( c.numRows() == 1 )
      c = c.transpose();
    setCols( X, i, i+1, c );
  }


  /**
   * Find the sum of a vector
   */
  public static double sum(double[] x ) {
    double sum = 0.0;
    for( int i = 0; i < x.length; i++ )
      sum += x[i];
    return sum;
  }
  public static double sum(int[] x ) {
    double sum = 0.0;
    for( int i = 0; i < x.length; i++ )
      sum += x[i];
    return sum;
  }

  /**
   * Find the sum of a vector
   */
  public static void normalize(double[] x) {
    double sum = sum( x );
    for( int i = 0; i < x.length; i++ )
      x[i] /= sum;
  }

  /**
   * Normalize the rows of X to lie between -1 and 1;
   */
  public static Pair<double[],double[]> rowCenter(double[][] X) {
    int rows = X.length;
    double[] rowMin = new double[rows];
    double[] rowMax = new double[rows];
    for( int row = 0; row < X.length; row++ ) {
      double rMin = rowMin[row] = min( X[row] );
      double rMax = rowMax[row] = max( X[row] );
      for( int i = 0; i < X[row].length; i++ )
        if( rMax > rMin )
          X[row][i] = 2 * (X[row][i] - rMin) / (rMax - rMin) - 1;
      else
          X[row][i] = 0;
    }
    return new Pair<>(rowMin, rowMax);
  }

  /**
   * Normalize the rows of X
   * @param X
   * @return normalized X, row minimums and row maximums
   */
  public static Triplet<SimpleMatrix,SimpleMatrix,SimpleMatrix> rowCenter(SimpleMatrix X) {
    double[][] X_ = MatrixFactory.toArray(X);
    Pair<double[], double[]> minMax = rowCenter(X_);
    return new Triplet<>(
            new SimpleMatrix(X_),
            MatrixFactory.fromVector(minMax.getValue0()),
            MatrixFactory.fromVector(minMax.getValue1()));
  }

  /**
   * Normalize the rows of X to lie between -1 and 1;
   */
  public static double[] rowScale(double[][] X) {
    int rows = X.length;
    double[] rowScale = new double[rows];
    for( int row = 0; row < X.length; row++ ) {
      double scale = rowScale[row] = maxAbs( X[row] );
      for( int i = 0; i < X[row].length; i++ )
          X[row][i] = X[row][i] / scale;
    }
    return rowScale;
  }

  /**
   * Normalize the rows of X
   * @param X
   * @return normalized X, row minimums and row maximums
   */
  public static Pair<SimpleMatrix,SimpleMatrix> rowScale(SimpleMatrix X) {
    double[][] X_ = MatrixFactory.toArray(X);
    double[] scaling = rowScale(X_);
    return new Pair<>( new SimpleMatrix(X_),
          MatrixFactory.fromVector(scaling) );
  }
  public static Pair<SimpleMatrix,SimpleMatrix> columnScale(SimpleMatrix X) {
    Pair<SimpleMatrix,SimpleMatrix> scaleInfo = rowScale(X.transpose());
    return new Pair<>(scaleInfo.getValue0().transpose(), scaleInfo.getValue1());
  }


  /**
   * Find the norm of a vector
   */
  public static double norm(double[] x) {
    double sum = 0.0;
    for( int i = 0; i < x.length; i++ )
      sum += x[i]*x[i];
    return Math.sqrt( sum );
  }

  /**
   * Find the norm of a vector
   */
  public static void makeUnitVector(double[] x) {
    double sum = norm( x );
    for( int i = 0; i < x.length; i++ )
      x[i] /= sum;
  }

  /**
   * Find the sum of the rows of the column in X 
   */
  public static double rowSum(DenseMatrix64F X, int row ) {
    double sum = 0;
    double[] X_ = X.data;
    for( int col = 0; col < X.numCols; col++ )
      sum += X_[ X.getIndex( row, col )];
    return sum;
  }
  /**
   * Find the sum of the entries of the column in X 
   */
  public static double rowSum(SimpleMatrix X, int row ) {
    return rowSum( X.getMatrix(), row );
  }

  /**
   * Find the sum of the entries of the column in X 
   */
  public static double columnSum(DenseMatrix64F X, int col ) {
    double sum = 0;
    double[] X_ = X.data;
    for( int row = 0; row < X.numRows; row++ )
      sum += X_[ X.getIndex( row, col )];
    return sum;
  }
  /**
   * Find the sum of the entries of the column in X 
   */
  public static double columnSum(SimpleMatrix X, int col ) {
    return columnSum( X.getMatrix(), col );
  }

  /**
   * Normalize a column of X
   */
  public static void columnNormalize(DenseMatrix64F X, int col ) {
    double sum = columnSum( X, col );
    double[] X_ = X.data;
    for( int row = 0; row < X.numRows; row++ )
      X_[ X.getIndex( row, col )] /= sum;
  }
  /**
   * Normalize a row of X
   */
  public static void rowNormalize(DenseMatrix64F X, int row ) {
    double sum = rowSum( X, row );
    double[] X_ = X.data;
    for( int col = 0; col < X.numCols; col++ )
      X_[ X.getIndex( row, col )] /= sum;
  }

  /**
   * Find the pairwise distance of the i-th row in X and j-th row in Y
   * @param X
   * @param Y
   * @return
   */
  public static DenseMatrix64F cdist(DenseMatrix64F X, DenseMatrix64F Y) {
    assert( X.numCols == Y.numCols );

    int n = X.numRows;
    int m = Y.numRows;

    DenseMatrix64F Z = new DenseMatrix64F( n, m );

    for( int i = 0; i < n; i++ ) {
      for( int j = 0; j < m; j++ ) {
        // Find the distance between X and Y
        double d = 0;
        for( int k = 0; k < X.numCols; k++ ) {
          double d_ = X.get(i,k) - Y.get(j,k);
          d += d_ * d_;
        }
        Z.set( Z.getIndex(i, j), Math.sqrt( d ) );
      }
    }

    return Z;
  }
  /**
   * Find the pairwise distance of the i-th row in X and j-th row in Y
   * @param X
   * @param Y
   * @return
   */
  public static SimpleMatrix cdist(SimpleMatrix X, SimpleMatrix Y) {
    return SimpleMatrix.wrap( cdist( X.getMatrix(), Y.getMatrix() ) );
  }


  /**
   * Project each column of X onto a simplex in place
   * @param x
   * @return
   */
  public static void projectOntoSimplex( double[] x ) {
    // Normalize and shrink to 0
    normalize(x);
    for(int i = 0; i < x.length; i++ )
      if( x[i] < 0 ) x[i] = 0;
    normalize(x);
  }

  public static void projectOntoSimplex( DenseMatrix64F X ) {
    int nRows = X.numRows;
    int nCols = X.numCols;

    double[] X_ = X.data;

    for( int col = 0; col < nCols; col++ ) {
      // For each column, normalize the vector and zero out negative values.
      columnNormalize( X, col );

      for( int row = 0; row < nRows; row++ ) {
        double x  = X_[ X.getIndex(row, col) ];
        if( x < 0 ) X_[ X.getIndex(row, col) ] = 0;
      }

      columnNormalize( X, col );
    }
  }

  /**
   * Project each columns of X onto a simplex
   * @param X
   * @return
   */
  public static SimpleMatrix projectOntoSimplex( SimpleMatrix X ) {
    DenseMatrix64F Y = X.getMatrix().copy();
    projectOntoSimplex(Y);
    return SimpleMatrix.wrap( Y );
  }	

  /**
   * Find the rank of the matrix
   */
  public static int rank( SimpleMatrix X, double eps ) {
    // HACK: X.svd().rank() does not give the right rank for some reason.
    SimpleMatrix W = X.svd().getW();
    int upperBound = ( W.numRows() < W.numCols() ) ? W.numRows() : W.numCols();

    for( int i = 0; i < upperBound; i++ )
      if( W.get( i, i ) < eps ) return i;

    return upperBound;
  }

  public static int rank( SimpleMatrix X ) {
    return rank( X, EPS_ZERO );
  }

  /**
   * Compute the SVD and compress it to choose the top k singular vectors
   */
  public static SimpleMatrix[] svdk( SimpleMatrix X, int k ) {
    @SuppressWarnings("unchecked")
    SimpleSVD<SimpleMatrix> UWV = X.svd(false);
    SimpleMatrix U = UWV.getU();
    SimpleMatrix W = UWV.getW();
    SimpleMatrix V = UWV.getV();

    // Truncate U, W and V to k-rank
    U = U.extractMatrix(0, SimpleMatrix.END, 0, k);
    W = W.extractMatrix(0, k, 0, k);
    V = V.extractMatrix(0, SimpleMatrix.END, 0, k);

    SimpleMatrix[] ret = {U, W, V};

    return ret;
  }

  /**
   * Compute the best k-rank approximation of the SVD
   */
  public static SimpleMatrix approxk( SimpleMatrix X, int k ) {
    SimpleMatrix[] UWV = svdk( X, k );

    SimpleMatrix U_ = UWV[0];
    SimpleMatrix W_ = UWV[1];
    SimpleMatrix V_ = UWV[2];

    return U_.mult( W_ ).mult( V_.transpose() );
  }

  /**
   * Compute eigenvalues and eigenvectors of X
   */
  public static SimpleMatrix[] eig( SimpleMatrix X ) throws NumericalException {
    SimpleMatrix L = new SimpleMatrix( 1, X.numCols() );
    SimpleMatrix R = new SimpleMatrix( X.numRows(), X.numCols() );

    @SuppressWarnings("unchecked")
    SimpleEVD<SimpleMatrix> EVD = X.eig();

    // Get the eigenvector matrix
    for( int i = 0; i<X.numRows(); i++ )
    {
      if( !EVD.getEigenvalue(i).isReal() )
        throw new NumericalException("Imaginary eigenvalue at index " + i);
      L.set( 0, i, EVD.getEigenvalue(i).real );
      MatrixOps.setCol( R, i, EVD.getEigenVector(i));
    }

    SimpleMatrix[] LR = {L, R};
    return LR;
  }

  /**
   * Get the sqrt of a diagonal matrix
   * @param X
   * @return
   */
  public static SimpleMatrix sqrt( SimpleMatrix X ) {
    double[] diagonal = MatrixFactory.toVector(X.extractDiag());
    for(int i = 0; i < diagonal.length; i++) {
      diagonal[i] = Math.sqrt(diagonal[i]);
    }
    return MatrixFactory.diag(MatrixFactory.fromVector(diagonal));
  }


  public static SimpleMatrix whitener( SimpleMatrix X ) {
    SimpleSVD UDV = X.svd();
    SimpleMatrix U = UDV.getU();
    SimpleMatrix D = UDV.getW();
    SimpleMatrix Dsqrtinv = sqrt( D ).invert();
//    return Dsqrtinv.mult(U.transpose());
    return U.mult(Dsqrtinv);
    //return (U.mult( sqrt( W ) )).invert();
  }
  public static SimpleMatrix colorer( SimpleMatrix X ) {
    SimpleSVD UWV = X.svd();
    SimpleMatrix U = UWV.getU();
    SimpleMatrix W = UWV.getW();
    return U.mult( sqrt( W ) );
  }

  /**
   * Align the rows of matrix X so that the rows/columns are matched with the
   * columns of Y
   */
  public static SimpleMatrix alignMatrix( SimpleMatrix X, SimpleMatrix Y ) {
    assert( X.numRows() == Y.numRows() );
    assert( X.numCols() == Y.numCols() );
    int nRows = X.numRows(); 
    int nCols = X.numCols(); 

    SimpleMatrix X_ = new SimpleMatrix( nRows, nCols );
    // Populate the weight matrix 
    double[][] W = MatrixFactory.toArray( cdist( Y, X ) );
    // Compute min-weight matching
    int[][] matching = HungarianAlgorithm.findWeightedMatching( W, false );
    // Shuffle rows
    for( int[] match : matching )
      setRow( X_, match[0], row( X, match[1] ) );

    return X_;
  }
  public static SimpleMatrix alignMatrix( SimpleMatrix X, SimpleMatrix Y, boolean compareColumns ) {
    if( compareColumns )
      return alignMatrix( X.transpose(), Y.transpose() ).transpose();
    else
      return alignMatrix( X, Y );
  }

  // Algebraic operations
  /**
   * Find the sum of a vector
   */
  public static void plus(double[] x, double y ) {
    for( int i = 0; i < x.length; i++ )
      x[i] += y;
  }
  public static void plus(double[] x, double[] y ) {
    assert( x.length == y.length );
    for( int i = 0; i < x.length; i++ )
      x[i] += y[i];
  }

  public static void minus(double[] x, double y ) {
    for( int i = 0; i < x.length; i++ )
      x[i] -= y;
  }
  public static void minus(double[] x, double[] y ) {
    assert( x.length == y.length );
    for( int i = 0; i < x.length; i++ )
      x[i] -= y[i];
  }

  public static void log(double[] x) {
    for( int i = 0; i < x.length; i++ )
      x[i] = Math.log( x[i] );
  }
  public static void exp(double[] x) {
    for( int i = 0; i < x.length; i++ )
      x[i] = Math.exp( x[i] );
  }


  public static double logsumexp(double x, double y) {
    double min_ = ( x < y ) ? x : y;
    double max_ = ( x < y ) ? y : x;
    return Math.log( 1 + Math.exp( max_ - min_ ) ) + min_;
  }

  public static double logsumexp(final double[] x) {
    // Reduce
    double logsum = x[0];
    for( int i = 1; i < x.length; i++ ) {
      logsum = logsumexp( logsum, x[i] );
    }
    return logsum;
  }
}


