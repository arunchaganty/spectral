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

public class MatrixOps {

  /**
   * Test whether two matrices are within eps of each other
   */
  public static boolean allclose( DenseMatrix64F X1, DenseMatrix64F X2, double eps ) {
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
    return allclose( X1, X2, 1e-4 );
  }
  public static boolean allclose( SimpleMatrix X1, SimpleMatrix X2, double eps ) {
    return allclose( X1.getMatrix(), X2.getMatrix(), eps );
  }
  public static boolean allclose( SimpleMatrix X1, SimpleMatrix X2 ) {
    return allclose( X1.getMatrix(), X2.getMatrix() );
  }

  /**
   * Find the minimium value of the matrix X
   */
  public static double min( DenseMatrix64F X ) {
    double[] X_ = X.data;
		double min = Double.POSITIVE_INFINITY;
		for( int i = 0; i < X_.length; i++ ) 
				if( X_[i] < min ) min = X_[i];

    return min;
  }
  public static double min( SimpleMatrix X ) {
    return min( X.getMatrix() );
  }
  /**
   * Find the maximum value of the matrix X
   */
  public static double max( DenseMatrix64F X ) {
    double[] X_ = X.data;
    double max = Double.NEGATIVE_INFINITY;
		for( int i = 0; i < X_.length; i++ ) 
				if( X_[i] > max ) max = X_[i];

    return max;
  }
  public static double max( SimpleMatrix X ) {
    return max( X.getMatrix() );
  }



  /**
   * Compute the average outer product of each row of X1 and X2
   */
  public static DenseMatrix64F Pairs( DenseMatrix64F X1, DenseMatrix64F X2 ) {
    assert( X1.numRows == X2.numCols );

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
		if( r.numRows() == 1 )
			r = r.transpose();
    setRows( X, i, i+1, r );
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
		if( c.numCols() == 1 )
			c = c.transpose();
    setCols( X, i, i+1, c );
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
	 * @param X
	 * @return
	 */
	public static void projectOntoSimplex( DenseMatrix64F X ) {
		int nRows = X.numRows;
		int nCols = X.numCols;

    double[] X_ = X.data;

		for( int col = 0; col < nCols; col++ ) {
      // For each column, shift the value towards $\bar{X} - 1.0/n$, 
      // Note that if $X$ is already on the simplex, then this leaves x
      // unchanged.
      
      double X_bar = columnSum( X, col )/nRows;

			for( int row = 0; row < nRows; row++ ) {
				double x  = X_[ X.getIndex(row, col) ];

        // If X_bar < 0 then project onto the -1 simplex.
				if( X_bar < 0 ) x = -(x - X_bar - 1.0/nRows);
				else x = x - X_bar + 1.0/nRows;
        // If the x is still less than zero, remove it
				if( x < 0 ) x = 0;
				X_[X.getIndex(row, col)] = x;
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
  public static int rank( SimpleMatrix X ) {
    return X.svd().rank();
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
    SimpleMatrix L = new SimpleMatrix( X.numRows(), 1 );
    SimpleMatrix R = new SimpleMatrix( X.numRows(), X.numCols() );

    @SuppressWarnings("unchecked")
    SimpleEVD<SimpleMatrix> EVD = X.eig();

    // Get the eigenvector matrix
    for( int i = 0; i<X.numRows(); i++ )
    {
      if( !EVD.getEigenvalue(i).isReal() )
        throw new NumericalException("Imaginary eigenvalue at index " + i);
      L.set( 0, i, EVD.getEigenvalue(i).real );
      MatrixOps.setRow( R, i, EVD.getEigenVector(i));
    }

    SimpleMatrix[] LR = {L, R};
    return LR;
  }
}


