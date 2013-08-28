/**
 * learning.linalg
 * Arun Chaganty (chaganty@stanford.edu)
 *
 */

package learning.linalg;

import learning.linalg.SimpleTensor;
import learning.exceptions.NumericalException;

import org.ejml.alg.dense.mult.VectorVectorMult;
import org.ejml.factory.DecompositionFactory;
import org.ejml.factory.QRDecomposition;
import org.ejml.factory.SingularValueDecomposition;
import org.ejml.ops.CommonOps;
import org.ejml.ops.SpecializedOps;
import org.ejml.simple.SimpleMatrix;
import org.ejml.data.DenseMatrix64F;
import org.ejml.simple.SimpleSVD;
import org.ejml.simple.SimpleEVD;
import org.javatuples.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

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
  public static void printSize( DenseMatrix64F X ) {
    System.out.printf( "(%d, %d)\n", X.numRows, X.numCols );
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
    if( X == null ) return "null";
    String out = "";
    out += "{\n";
    for( int i = 0; i < X.length; i++ ) {
      out += "{ ";
      for( int j = 0; j < X[i].length; j++ )
        out += String.valueOf(X[i][j]) + ", ";
      out += "},\n";
    }
    out += "}";

    return out;
  }
  public static String arrayToString( double[][][] X ) {
    if( X == null ) return "null";
    String out = "";
    out += "{\n";
    for( int i = 0; i < X.length; i++ ) {
      out += "{ ";
      for( int j = 0; j < X[i].length; j++ ) {
        for( int k = 0; k < X[i][j].length; k++ ) 
          out += String.valueOf(X[i][j][k]) + ", ";
        out += "}, \n";
      }
      out += "},\n\n";
    }
    out += "}";

    return out;
  }
  public static String arrayToString( int[][] X ) {
    if( X == null ) return "null";
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

  public static void printArray( double[][][] X ) {
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
  public static double dot( double[] x, double[] y, boolean[] select ) {
    assert( x.length == y.length );
    double sum = 0.0;
    for( int i = 0; i < x.length; i++ )
      sum += (select[i]) ? x[i] * y[i] : 0;
    return sum;
  }
  public static double dot( DenseMatrix64F x, DenseMatrix64F y ) {
    return VectorVectorMult.innerProd( x, y );
  }

  public static double[][] mult( double[][] A, double[][] B ) {
    assert( A[0].length == B.length );
    int m = A.length;
    int n = A[0].length;
    int l = B[0].length;

    double[][] C = new double[m][l];
    for( int i = 0; i < m; i++ )
      for( int j = 0; j < l; j++ )
        for( int k = 0; k < l; k++ )
          C[i][j] += A[i][k] * B[k][j];

   return C;
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
  public static boolean allclose( double[][] X1, double[][] X2, double eps ) {
    assert( X1.length == X2.length );
    assert( X1[0].length == X2[0].length );

    for( int i = 0; i < X1.length; i++ ) {
      for( int j = 0; j < X1[0].length; j++ ) {
        if( !equal( X1[i][j], X2[i][j], eps ) ) return false;
      }
    }

    return true;
  }
  public static boolean allclose( double[][] X1, double[][] X2 ) {
    return allclose( X1, X2, EPS_CLOSE );
  }
  public static boolean allclose( double[][][] X1, double[][][] X2, double eps ) {
    assert( X1.length == X2.length );
    assert( X1[0].length == X2[0].length );
    assert( X1[0][0].length == X2[0][0].length );

    for( int i = 0; i < X1.length; i++ ) {
      for( int j = 0; j < X1[0].length; j++ ) {
        for( int k = 0; k < X1[0][0].length; k++ ) {
          if( !equal( X1[i][j][k], X2[i][j][k], eps ) ) return false;
        }
      }
    }

    return true;
  }
  public static boolean allclose( FullTensor X1, FullTensor X2, double eps ) {
    return allclose( X1.X, X2.X, eps );
  }
  public static boolean allclose( double[][][] X1, double[][][] X2 ) {
    return allclose( X1, X2, EPS_CLOSE );
  }
  public static boolean allclose( FullTensor X1, FullTensor X2 ) {
    return allclose( X1.X, X2.X );
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
  public static double norm(FullTensor T) {
    double norm = 0.0;
    for(int d1 = 0; d1 < T.D1; d1++ )
      for(int d2 = 0; d2 < T.D2; d2++ )
        for(int d3 = 0; d3 < T.D3; d3++ )
          norm += (T.X[d1][d2][d3]) * (T.X[d1][d2][d3]);
    return Math.sqrt(norm);
  }
  public static double diff(DenseMatrix64F X, DenseMatrix64F Y) {
    return SpecializedOps.diffNormF(X, Y);
  }
  public static double diff(SimpleMatrix X, SimpleMatrix Y) {
    return diff(X.getMatrix(), Y.getMatrix());
  }
  public static double diff(FullTensor X, FullTensor Y) {
    assert( X.D1 == Y.D1 ); assert( X.D2 == Y.D2 ); assert( X.D3 == Y.D3 );

    double err = 0.0;
    for(int d1 = 0; d1 < X.D1; d1++)
      for(int d2 = 0; d2 < X.D2; d2++)
        for(int d3 = 0; d3 < X.D3; d3++)
          err += (X.X[d1][d2][d3] - Y.X[d1][d2][d3]) * (X.X[d1][d2][d3] - Y.X[d1][d2][d3]);
    return Math.sqrt(err);
  }
  public static double maxdiff(FullTensor X, FullTensor Y) {
    assert( X.D1 == Y.D1 ); assert( X.D2 == Y.D2 ); assert( X.D3 == Y.D3 );

    double err = 0.0;
    for(int d1 = 0; d1 < X.D1; d1++)
      for(int d2 = 0; d2 < X.D2; d2++)
        for(int d3 = 0; d3 < X.D3; d3++)
          err = Math.max( err, (X.X[d1][d2][d3] - Y.X[d1][d2][d3]) );
    return err;
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
  public static double min( double x, double y ) {
    return ( x < y ) ? x : y;
  }
  public static int min( int x, int y ) {
    return ( x < y ) ? x : y;
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
  public static double max( double x, double y ) {
    return ( x > y ) ? x : y;
  }
  public static int max( int x, int y ) {
    return ( x > y ) ? x : y;
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
  public static int argmax( int[] x ) {
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
   * Extract the outer product of x
   */
  public static void outer( DenseMatrix64F x, DenseMatrix64F y, DenseMatrix64F XY ) {
    VectorVectorMult.outerProd(x, y, XY);
  }
  /**
   * Compute the average outer product of each row of X1 and X2
   */
  public static DenseMatrix64F Pairs( DenseMatrix64F X1, DenseMatrix64F X2 ) {
    // TODO: Optimize
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
//  public static SimpleTensor Triples( SimpleMatrix X1, SimpleMatrix X2, SimpleMatrix X3 ) {
//    return new SimpleTensor( X1, X2, X3 );
//  }
  public static FullTensor Triples( DenseMatrix64F X1, DenseMatrix64F X2, DenseMatrix64F X3 ) {
    // TODO: Optimize
    assert( X1.numRows == X2.numRows && X2.numRows == X3.numRows );

    int nRows = X1.numRows;
    int d1 = X1.numCols;
    int d2 = X2.numCols;
    int d3 = X3.numCols;

    double[] X1_ = X1.data;
    double[] X2_ = X2.data;
    double[] X3_ = X3.data;
    double[][][] Z = new double[d1][d2][d3];

    // Average the outer products
    for(int row = 0; row < nRows; row++ ) {
      for( int i = 0; i < d1; i++ ) {
        double x1 = X1_[X1.getIndex(row, i)];
        for( int j = 0; j < d2; j++ ) {
          double x2 = X2_[X2.getIndex(row, j)];
          for( int k = 0; k < d3; k++ ) {
            double x3 = X3_[X3.getIndex(row, k)];
            // Rolling average
            Z[i][j][k] += (x1*x2*x3 - Z[i][j][k])/(row+1);
          }
        }
      }
    }

    return new FullTensor(Z);
  }
  public static FullTensor Triples( SimpleMatrix X1, SimpleMatrix X2, SimpleMatrix X3 ) {
    return Triples(X1.getMatrix(), X2.getMatrix(), X3.getMatrix());
  }

  /**
   * Extract the i-th column of X
   */
  public static SimpleMatrix col( SimpleMatrix X, int col ) {
    return X.extractMatrix( 0, SimpleMatrix.END, col, col+1 );
  }
  public static void col( DenseMatrix64F src, int col, DenseMatrix64F dest ) {
    CommonOps.extract( src, 0, src.numRows, col, col+1, dest, 0, 0 );
  }
  public static DenseMatrix64F col( DenseMatrix64F X, int col ) {
    DenseMatrix64F x = new DenseMatrix64F( X.numRows, 1 );
    col(X, col, x);
    return x;
  }

  /**
   * Extract the i-th row of X
   */
  public static SimpleMatrix row( SimpleMatrix X, int row ) {
    return X.extractMatrix( row, row+1, 0, SimpleMatrix.END);
  }
  public static void row( DenseMatrix64F src, int row, DenseMatrix64F dest ) {
    CommonOps.extract( src, row, row+1, 0, src.numCols, dest, 0, 0 );
  }
  public static DenseMatrix64F row( DenseMatrix64F X, int row ) {
    DenseMatrix64F x = new DenseMatrix64F( 1, X.numCols );
    row(X, row,  x);
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
  public static double sum(boolean[] x ) {
    double sum = 0.0;
    for( int i = 0; i < x.length; i++ )
      sum += x[i] ? 1 : 0;
    return sum;
  }
  public static double sum(double[][] x ) {
    double sum = 0.0;
    for( int i = 0; i < x.length; i++ )
      for( int j = 0; j < x[i].length; j++ )
        sum += x[i][j];
    return sum;
  }
  // 0 for rows, 1 for columns
  public static double sum(double[][] x, int axis, int index ) {
    // Assume symmetric.
    double sum = 0.0;
    if( axis == 0 ) {
      for( int j = 0; j < x[index].length; j++ )
        sum += x[index][j];
    } else {
      for( int i = 0; i < x.length; i++ )
          sum += x[i][index];
    }
    return sum;
  }
  public static double sum(double[][][] x, int axis, int index ) {
    // Assume symmetric.
    double sum = 0.0;
    if( axis == 0 ) {
      for( int i = 0; i < x[index].length; i++ )
        for( int j = 0; j < x[index][i].length; j++ )
          sum += x[index][i][j];
    } else if( axis == 1) {
      for( int i = 0; i < x.length; i++ )
        for( int j = 0; j < x[i][index].length; j++ )
          sum += x[i][index][j];
    } else {
      for( int i = 0; i < x.length; i++ )
        for( int j = 0; j < x[i].length; j++ )
          sum += x[i][j][index];
    }
    return sum;
  }
  public static double sum(double[][][] x, int axis1, int index1, int axis2, int index2  ) {
    // Assume symmetric.
    assert( axis1 < axis2 );
    double sum = 0.0;
    if( axis1 == 0 ) {
      if( axis2 == 1 )  {
        for( int i = 0; i < x[index1][index2].length; i++ )
          sum += x[index1][index2][i];
      } else {
        for( int i = 0; i < x[index1].length; i++ )
          sum += x[index1][i][index2];
      }
    } else if( axis1 == 1) {
      assert(axis2 == 2);
      for( int i = 0; i < x.length; i++ )
          sum += x[i][index1][index2];
    }
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
  public static SimpleMatrix normalize(SimpleMatrix x) {
    assert( isVector( x ) );
    return x.scale( 1.0 / x.elementSum() );
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
  public static void makeUnitVector(DenseMatrix64F X) {
    makeUnitVector(X.data);
  }
  public static SimpleMatrix makeUnitVector(SimpleMatrix X) {
    DenseMatrix64F Y = X.getMatrix().copy();
    makeUnitVector(Y);
    return SimpleMatrix.wrap(Y);
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
  public static void projectOntoSimplex( double[] x, double smooth ) {
    // Normalize and shrink to 0
    normalize(x);
    for(int i = 0; i < x.length; i++ ) {
      if( x[i] < 0 ) x[i] = 0;
      x[i] += smooth;
    }
    normalize(x);
  }
  public static void projectOntoSimplex( double[] x ) {
    projectOntoSimplex( x, 0.0 );
  }

  public static void projectOntoSimplex( DenseMatrix64F X, double smooth ) {
    int nRows = X.numRows;
    int nCols = X.numCols;

    double[] X_ = X.data;

    for( int col = 0; col < nCols; col++ ) {
      // For each column, normalize the vector and zero out negative values.
      // Get the majority sign
      int positives = 0;
      for( int row = 0; row < nRows; row++ ) positives += (X.get(row,col) > 0) ? 1 : 0;

      // Flip sign
      if( positives < nRows / 2 ) for( int row = 0; row < nRows; row++ )
        X.set(row, col, -X.get(row,col));

      for( int row = 0; row < nRows; row++ ) {
        double x  = X_[ X.getIndex(row, col) ];
        if( x < 0 ) X_[ X.getIndex(row, col) ] = 0;
        X_[ X.getIndex(row, col) ] += smooth;
      }

      columnNormalize( X, col );
    }
  }
  public static void projectOntoSimplex( DenseMatrix64F X ) {
    projectOntoSimplex( X, 0.0 );
  }

  /**
   * Project each columns of X onto a simplex
   * @param X
   * @return
   */
  public static SimpleMatrix projectOntoSimplex( SimpleMatrix X, double smooth ) {
    DenseMatrix64F Y = X.getMatrix().copy();
    projectOntoSimplex(Y, smooth);
    return SimpleMatrix.wrap( Y );
  }	
  public static SimpleMatrix projectOntoSimplex( SimpleMatrix X ) {
    return projectOntoSimplex( X, 0.0 );
  }	

  /**
   * Find the rank of the matrix
   */
  public static int rank( DenseMatrix64F X, double eps ) {
    return rank( SimpleMatrix.wrap(X), eps );
  }
  public static int rank( DenseMatrix64F X ) {
    return rank( X, EPS_ZERO );
  }
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
  public static Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> svdk( SimpleMatrix X, int K ) {
    @SuppressWarnings("unchecked")
    SimpleSVD<SimpleMatrix> UWV = X.svd(false);
    SimpleMatrix U = UWV.getU();
    SimpleMatrix W = UWV.getW();
    SimpleMatrix V = UWV.getV();

    // Truncate U, W and V to k-rank
    U = U.extractMatrix(0, SimpleMatrix.END, 0, K);
    W = W.extractMatrix(0, K, 0, K);
    V = V.extractMatrix(0, SimpleMatrix.END, 0, K);

    return new Triplet<>(U, W, V);
  }
  public static Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> svdk( SimpleMatrix X ) {
    @SuppressWarnings("unchecked")
    SimpleSVD<SimpleMatrix> UWV = X.svd(false);
    int K = UWV.rank();
    SimpleMatrix U = UWV.getU();
    SimpleMatrix W = UWV.getW();
    SimpleMatrix V = UWV.getV();

    // Truncate U, W and V to k-rank
    U = U.extractMatrix(0, SimpleMatrix.END, 0, K);
    W = W.extractMatrix(0, K, 0, K);
    V = V.extractMatrix(0, SimpleMatrix.END, 0, K);

    return new Triplet<>(U, W, V);
  }
  public static Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> svdk( DenseMatrix64F X ) {
    return svdk(SimpleMatrix.wrap(X));
  }
  public static Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> svd( SimpleMatrix X ) {
    @SuppressWarnings("unchecked")
    SimpleSVD<SimpleMatrix> UWV = X.svd(true);
    SimpleMatrix U = UWV.getU();
    SimpleMatrix W = UWV.getW();
    SimpleMatrix V = UWV.getV();
    return new Triplet<>(U, W, V);
  }

  /**
   * Compute the best k-rank approximation of the SVD
   */
  public static SimpleMatrix approxk( SimpleMatrix X, int k ) {
    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> UWV = svdk( X, k );

    SimpleMatrix U_ = UWV.getValue0();
    SimpleMatrix W_ = UWV.getValue1();
    SimpleMatrix V_ = UWV.getValue2();

    return U_.mult( W_ ).mult( V_.transpose() );
  }

  /**
   * This is a weak approximation, as rank-mode is an upper bound on true rank
   * @param X
   * @param K
   */
  public static void approxk( FullTensor X, int K ) {
    for( int i = 0; i < 3; i++ ) {
      SimpleMatrix Y = X.unfold(i);
      Y = MatrixOps.approxk(Y, K);
      FullTensor.fold(i, Y.getMatrix(), X);
    }
  }

  public static double conditionNumber( SimpleMatrix X, int k ) {
    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> UDV = svdk( X, k );
    SimpleMatrix D = UDV.getValue1();
    double[] diagonal = MatrixFactory.toVector(D.extractDiag());
    return diagonal[0]/diagonal[k-1];
  }

  public static double sigmak( SimpleMatrix X, int k ) {
    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> UDV = svdk( X, k );
    SimpleMatrix D = UDV.getValue1();
    double[] diagonal = MatrixFactory.toVector(D.extractDiag());
    return diagonal[k-1];
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

  public static SimpleMatrix whitener( SimpleMatrix X, int K ) {
    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> UDV = svdk(X, K);

    SimpleMatrix U = UDV.getValue0();
    SimpleMatrix D = UDV.getValue1();
    SimpleMatrix Dsqrtinv = sqrt( D ).invert();
    return U.mult(Dsqrtinv);
  }
  public static SimpleMatrix whitener( SimpleMatrix X ) {
    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> UDV = svdk(X);

    SimpleMatrix U = UDV.getValue0();
    SimpleMatrix D = UDV.getValue1();
    return U.mult(sqrt( D ).invert());
  }
  public static SimpleMatrix randomizedWhitener( SimpleMatrix X, SimpleMatrix Q, int K ) {
    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> UDV = svd(X);
    SimpleMatrix U = UDV.getValue0();
    U = Q.mult(U);
    SimpleMatrix D = UDV.getValue1();
    // Truncate
    U = U.extractMatrix(0, SimpleMatrix.END, 0, K);
    D = D.extractMatrix(0, K, 0, K);
    SimpleMatrix Dsqrtinv = sqrt( D ).invert();
    return U.mult(Dsqrtinv);
  }
  public static SimpleMatrix colorer( SimpleMatrix X ) {
    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> UDV = svdk(X);
    SimpleMatrix U = UDV.getValue0();
    SimpleMatrix D = UDV.getValue1();
    return U.mult(sqrt(D));
  }
  public static SimpleMatrix colorer( SimpleMatrix X, int K ) {
    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> UDV = svdk(X, K);
    SimpleMatrix U = UDV.getValue0();
    SimpleMatrix D = UDV.getValue1();
    return U.mult(sqrt(D));
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
  public static int[] alignRows( SimpleMatrix X, SimpleMatrix Y) {
    assert( X.numRows() == Y.numRows() );
    assert( X.numCols() == Y.numCols() );
    int nRows = X.numRows();
    int nCols = X.numCols();

    // Populate the weight matrix
    double[][] W = MatrixFactory.toArray( cdist( Y, X ) );
    // Compute min-weight matching
    int[][] matching = HungarianAlgorithm.findWeightedMatching( W, false );
    // Shuffle rows
    int[] perm = new int[nRows];
    for( int[] match : matching )
      perm[match[0]] = match[1];

    return perm;
  }
  public static int[] alignColumns( SimpleMatrix X, SimpleMatrix Y) {
    return alignRows( X.transpose(), Y.transpose() );
  }

  public static SimpleMatrix permuteRows( SimpleMatrix X, int[] perm ) {
    int nRows = X.numRows();
    int nCols = X.numCols();
    SimpleMatrix X_ = new SimpleMatrix( nRows, nCols );
    // Shuffle rows
    for( int i = 0; i < nRows; i++ ) {
      setRow( X_, i, row( X, perm[i] ) );
    }

    return X_;
  }
  public static SimpleMatrix permuteColumns( SimpleMatrix X, int[] perm ) {
    int nRows = X.numRows();
    int nCols = X.numCols();
    SimpleMatrix X_ = new SimpleMatrix( nRows, nCols );
    // Shuffle rows
    for( int i = 0; i < nCols; i++ ) {
      setCol( X_, i, col( X, perm[i] ) );
    }

    return X_;
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
    return Math.log( 1 + Math.exp( min_ - max_ ) ) + max_;
  }

  public static double logsumexp(final double[] x) {
    // Reduce
    double logsum = x[0];
    for( int i = 1; i < x.length; i++ ) {
      logsum = logsumexp( logsum, x[i] );
    }
    return logsum;
  }

  /**
   * Compute the quadratic form: y_i = x_i^T M x_i
   * @param x
   * @param M
   * @param y
   */
  public static double xMy(final DenseMatrix64F x, final DenseMatrix64F M, final DenseMatrix64F y) {
    return VectorVectorMult.innerProdA(x, M, y);
  }
  public static void quadraticForm(final DenseMatrix64F X, final DenseMatrix64F M, DenseMatrix64F y) {
    int N = X.numRows;
    int D = X.numCols;

    DenseMatrix64F x = new DenseMatrix64F(1, D);

    for( int n = 0; n < N; n++ ) {
      row( X, n, x );
      double v = VectorVectorMult.innerProdA(x, M, x);
      y.set( n, v );
    }
  }
  public static SimpleMatrix quadraticForm(SimpleMatrix X, SimpleMatrix M) {
    DenseMatrix64F y = new DenseMatrix64F(X.numRows(), 1);
    quadraticForm(X.getMatrix(), M.getMatrix(), y);
    return SimpleMatrix.wrap(y);
  }

  /**
   * Do the incremental weighted averging step
   * (X <- (w * dX - X)/(n+1))
   * @param n
   * @param w
   * @param dX
   * @param X
   */
  public static void incrementalAverageUpdate( double w, int n, DenseMatrix64F dX, DenseMatrix64F X) {

    if( w != 1.0 )
      CommonOps.scale(w, dX);
    CommonOps.subEquals(dX, X);
    CommonOps.scale(1.0 / (n + 1), dX);
    CommonOps.addEquals(X, dX);
  }
  public static void incrementalAverageUpdate( int n, DenseMatrix64F dX, DenseMatrix64F X) {
    incrementalAverageUpdate(1.0, n, dX, X);
  }

  public static double[][] removeInRange( double[][] X, double lbound, double ubound ) {
    ArrayList<double[]> entries = new ArrayList<>();
    for( double[] x : X ) {
      boolean inRange = false;
      for( double x_i : x ) if( x_i  > lbound && x_i < ubound ) inRange = true;
      if(!inRange)
        entries.add(x);
    }

    double[][] Y = entries.toArray(new double[][]{});
    return Y;
  }
  public static SimpleMatrix removeInRange( SimpleMatrix X, double lbound, double ubound ) {
    return new SimpleMatrix( removeInRange( MatrixFactory.toArray(X), lbound, ubound ) );
  }

  public static boolean isSymmetric( SimpleMatrix M ) {
    if( M.numRows() != M.numCols() ) return false;
    int D = M.numRows();

    for( int d = 0; d < D; d++ )
      for( int d_ = 0; d_ <= d; d_++ )
        if( !equal(M.get( d, d_ ), M.get( d_, d ) ) )
          return false;

    return true;
  }

  public static boolean isSymmetric( FullTensor T ) {
    if( ! (T.D1 == T.D2 && T.D2 == T.D3 ) ) return false;
    int D = T.D1;

    for( int d1 = 0; d1 < D; d1++ ) {
      for( int d2 = 0; d2 < D; d2++ ) {
        for( int d3 = 0; d3 < D; d3++ ) {
          boolean isSymmetric = 
            equal(T.X[d1][d2][d3], T.X[d1][d3][d2]) &&
            equal(T.X[d1][d2][d3], T.X[d2][d1][d3]) &&
            equal(T.X[d1][d2][d3], T.X[d2][d3][d1]) &&
            equal(T.X[d1][d2][d3], T.X[d3][d1][d2]) &&
            equal(T.X[d1][d2][d3], T.X[d3][d2][d1]);
          if(!isSymmetric) return false;
        }
      }
    }

    return true;
  }
  public static double symmetricSkewMeasure( FullTensor T ) {
    double skew = 0.0;
    assert(T.D1 == T.D2 && T.D2 == T.D3 );
    int D = T.D1;

    for( int d1 = 0; d1 < D; d1++ ) {
      for( int d2 = 0; d2 < D; d2++ ) {
        for( int d3 = 0; d3 < D; d3++ ) {
          skew += ( Math.abs(T.X[d1][d2][d3] - T.X[d1][d3][d2]) +
                    Math.abs(T.X[d1][d2][d3] - T.X[d2][d1][d3]) +
                    Math.abs(T.X[d1][d2][d3] - T.X[d2][d3][d1]) +
                    Math.abs(T.X[d1][d2][d3] - T.X[d3][d1][d2]) +
                    Math.abs(T.X[d1][d2][d3] - T.X[d3][d2][d1]) ) / 5;
        }
      }
    }

    return skew;
  }

  /**
   * Computes the reciprocal of a vector;
   * w_{ii} <- 1/w_{ii}
   */
  public static void reciprocal( DenseMatrix64F X ) {
    assert( isVector(X) );
    int D = (X.numRows == 1) ? X.numCols : X.numRows;
    for( int d = 0; d < D; d++ )
      X.set( d, 1/X.get(d) );
  }
  public static SimpleMatrix reciprocal( SimpleMatrix X ) {
    DenseMatrix64F Y = X.getMatrix().copy();
    reciprocal(Y);
    return SimpleMatrix.wrap( Y ) ;
  }

  /**
   * x[i] <- x[i] + y[i].
   */
  public static void add(double[] x, double[] y) {
    assert( x.length == y.length );

    for( int i = 0; i < x.length; i++ )
      x[i] += y[i];
  }

  /**
   * Report hamming loss between labels x, y
   */
  public static int hamming(int[] x, int[] y) {
    assert( x.length == y.length );
    int err = 0;
    for( int i = 0; i < x.length; i++ )
      err += (x[i] != y[i]) ? 1 : 0;
    return err / x.length;
  }

  public static void scale( double[][] x, double factor ) {
    for( int i = 0; i < x.length; i++ )
      for( int j = 0; j < x[i].length; j++ )
        x[i][j] *= factor;
  }
  public static void scale( double[] x, double factor ) {
    for( int i = 0; i < x.length; i++ )
        x[i] *= factor;
  }

  /**
   * Returns the indicies of decreasing values of col
   * @param col
   * @return
   */
  public static Integer[] argsort(final SimpleMatrix col) {
    Integer[] indices = new Integer[col.getNumElements()];
    for( int i = 0; i < indices.length; i++ ) indices[i] = i;

    Arrays.sort(indices, new Comparator<Integer>() {
      @Override
      public int compare(Integer i1, Integer i2) {
        return -Double.compare(col.get(i1), col.get(i2));
      }
    });
    // Make sure it's in descending order
    assert( col.get(indices[0]) > col.get(indices[indices.length-1]) );

    return indices;
  }

  public static interface Matrixable {
    public int numRows();
    public int numCols();
    // TODO: Change everything to be SimpleMatrix?
    public SimpleMatrix rightMultiply(SimpleMatrix right);
    public SimpleMatrix leftMultiply(SimpleMatrix left);
    public SimpleMatrix doubleMultiply(SimpleMatrix left, SimpleMatrix right);
  }
  public static Matrixable matrixable(final SimpleMatrix M) {
    return new Matrixable() {

      @Override
      public int numRows() {
        return M.numRows();
      }

      @Override
      public int numCols() {
        return M.numCols();
      }

      @Override
      public SimpleMatrix rightMultiply(SimpleMatrix right) {
        return M.mult(right);
      }

      @Override
      public SimpleMatrix leftMultiply(SimpleMatrix leftT) {
        return leftT.transpose().mult(M);
      }

      @Override
      public SimpleMatrix doubleMultiply(SimpleMatrix leftT, SimpleMatrix right) {
        return leftT.transpose().mult(M).mult(right);
      }
    };
  }

  public static interface Tensorable {
    public int numD1();
    public int numD2();
    public int numD3();
    // TODO: Implement elsewhere
//    public FullTensor multiply1(SimpleMatrix M);
//    public FullTensor multiply2(SimpleMatrix M);
//    public FullTensor multiply12(SimpleMatrix M, SimpleMatrix N);
    FullTensor multiply123(SimpleMatrix L, SimpleMatrix M, SimpleMatrix N);
  }
  public static Tensorable tensorable(final FullTensor T) {
    return new Tensorable() {
      @Override
      public int numD1() {
        return T.D1;
      }

      @Override
      public int numD2() {
        return T.D2;
      }

      @Override
      public int numD3() {
        return T.D3;
      }

      // TODO: Implement elsewhere
//      @Override
//      public FullTensor multiply1(SimpleMatrix M) {
//        return T.rotate(0, M);
//      }
//
//      @Override
//      public FullTensor multiply2(SimpleMatrix M) {
//        return T.rotate(1, M);
//      }
//
//      @Override
//      public FullTensor multiply12(SimpleMatrix M, SimpleMatrix N) {
//        return T.rotate(M, N, SimpleMatrix.identity(T.D3));
//      }
      @Override
      public FullTensor multiply123(SimpleMatrix L, SimpleMatrix M, SimpleMatrix N) {
        return T.rotate(L, M, N);
      }
    };
  }


  /**
   * Implements the randomized range finder routine from:
   *   Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions
   *   Nathan Halko, Per-Gunnar Martinsson, Joel A. Tropp
   *   http://arxiv.org/pdf/0909.4061
   *
   * @param M - object supporting multiply routines (used for random projections)
   * @param p - Size of the random projection. Should be about $2k$, where $k$ is the rank approximation of $M$ you desire
   * @param rnd - Random generator
   * @return - the range of M, Q (i.e. A ~= Q Q* A)
   */
  public static SimpleMatrix randomizedRangeFinder(Matrixable M, int p, Random rnd) {
    // Create a random matrix
    int n = M.numRows();
    SimpleMatrix Omega = RandomFactory.randn(rnd, n, p);

    // Form the product Y = A Omega
    DenseMatrix64F Y = M.rightMultiply(Omega).getMatrix();
    QRDecomposition<DenseMatrix64F> qr = DecompositionFactory.qr(n, p);
    qr.decompose(Y);
    DenseMatrix64F Q = qr.getQ(null, true);

    return SimpleMatrix.wrap(Q);
  }

  /**
   * Compute the SVD approximately.
   * @param M - object supporting matrix multiply routines
   * @param Qt - range of M
   * @return - SVD of M
   */
  public static Triplet<SimpleMatrix,SimpleMatrix,SimpleMatrix> randomizedSvd(Matrixable M, SimpleMatrix Qt) {
    int N = M.numRows();

    // Form B = Q* A
    SimpleMatrix B = new SimpleMatrix(M.leftMultiply(Qt));
    // Compute SVD of smaller matrix
    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> UDV = svd(B);

    // Project up: $U = Q\tilde{U}$
    SimpleMatrix U = (new SimpleMatrix(Qt)).mult(UDV.getValue0());

    return Triplet.with(U, UDV.getValue1(), UDV.getValue2());
  }
  public static Triplet<SimpleMatrix,SimpleMatrix,SimpleMatrix> randomizedSvd(Matrixable M, SimpleMatrix Qt, int K) {
    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> UWV = randomizedSvd(M, Qt);
    SimpleMatrix U = UWV.getValue0();
    SimpleMatrix W = UWV.getValue1();
    SimpleMatrix V = UWV.getValue2();

    // Truncate U, W and V to k-rank
    U = U.extractMatrix(0, SimpleMatrix.END, 0, K);
    W = W.extractMatrix(0, K, 0, K);
    V = V.extractMatrix(0, SimpleMatrix.END, 0, K);

    return Triplet.with(U, W, V);
  }
}

