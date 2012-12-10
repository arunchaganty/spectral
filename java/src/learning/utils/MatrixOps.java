package learning.utils;

/**
 * General vector operations
 */
public class MatrixOps {

  /*
	 * Computes the Frobenius norm
	 * @param x
	 * @return
	 */
	public static double norm( double[][] x ) {
		int nRows = x.length;
		int nCols = x[0].length;
		
		double result = 0.0;
		for( int i = 0; i < nRows; i++ ) {
			for( int j = 0; j < nCols; j++ )
				result += x[i][j] * x[i][j];
		}
		return Math.sqrt(result);
	}
	
	/**
	 * Transpose the matrix x
	 * @param x
	 * @return
	 */
	public static double[][] transpose( double[][] x ) {
		int nRows = x.length;
		int nCols = x[0].length;
		
		double[][] result = new double[nCols][nRows];
		for( int i = 0; i < nCols; i++ ) {
			for( int j = 0; j < nRows; j++ )
				result[i][j] = x[j][i];
		}
		return result;
	}
	
	/**
	 * Return the vector dot-product for x and y
	 * @param x
	 * @param y
	 * @return
	 */
	public static double dot( double[] x, double[] y ) {
		assert( x.length == y.length );
		double prod = 0.0;
		for( int i = 0; i < x.length; i++ )
			prod += x[i] * y[i];
		
		return prod;
	}
	
	/**
	 * Return the vector dot-product for x and y
	 * @param x
	 * @param y
	 * @return
	 */
	public static double[] dot( double[][] x, double[] y ) {
		int d = x.length;
		int r = x[0].length;
		assert( y.length == r );
		
		double[] result = new double[d];
		for( int i = 0; i < d; i++ ) {
			for( int j = 0; j < r; j++ )
				result[i] += x[i][j] * y[j];
		}
		return result;
	}
	
	/**
	 * Compute the sum of the entries of x
	 * @param x
	 * @return
	 */
	public static double sum( double[] x ) {
		double sum = 0.0;
		for( int i = 0; i < x.length; i++ )
			sum += x[i];
		return sum;
	}
	
	/**
	 * Compute the sum of the entries of x
	 * @param x
	 * @return
	 */
	public static double sum( double[][] x ) {
		double sum = 0.0;
		for( int i = 0; i < x.length; i++ )
			for( int j = 0; j < x[i].length; j++ )
				sum += x[i][j];
		return sum;
	}
	
	/**
	 * Compute a X + b Y
	 * @param x
	 * @return
	 */
	public static double[][] matrixAdd( double a, double[][] X, double b, double[][] Y ) {
    assert( X.length == Y.length );
    assert( X[0].length == Y[0].length );

    int r = X.length;
    int c = X[0].length;

		double[][] result = new double[r][c];
		for( int i = 0; i < r; i++ )
			for( int j = 0; j < c; j++ )
				result[i][j] = a * X[i][j] + b * Y[i][j];
		return result;
	}

	public static double[][] matrixAdd( double[][] X, double[][] Y ) {
    return matrixAdd( 1, X, 1, Y );
  }

	public static double[][] matrixSub( double[][] X, double[][] Y ) {
    return matrixAdd( 1, X, -1, Y );
  }

	/**
	 * Normalize the entries of x to sum to 1
	 * Note: Changes x in place
	 * @param x
	 */
	public static void normalize( double[] x ) {
		double sum = sum(x);
		for( int i = 0; i < x.length; i++ )
			x[i] /= sum;
	}
	
	/**
	 * Add a vector to each row/col of a matrix
	 * @param n
	 * @param M
	 * @return
	 */
	public static void vectorPlus( double[][] M, double[] v, boolean addToRows ) {
		int d = M.length;
		int r = M[0].length;
		
		if( addToRows ) {
			assert( v.length == r );
			for(int i = 0; i<d; i++ ) {
				for(int j =  0; j < r; j++)
					M[i][j] += v[j];
			}
		}
		else {
			assert( v.length == d );
			for(int i = 0; i < d; i++ ) {
				for(int j =  0; j < r; j++)
					M[i][j] += v[i];
			}
		}
	}
	
	/**
	 * Stack v n times to form a n x v matrix
	 * @param n
	 * @param M
	 * @return
	 */
	public static double[][] vectorStack( int n, double[] v ) {
		int d = v.length;
		double[][] result = new double[d][d];
		
		for( int i = 0; i < n; i++ )
			for( int j = 0; j < d; j++ )
				result[i][j] = v[j];
		return result;
	}
	
	/**
	 * Compute the empirical second moment
	 * @param X1 - Data with each point as a row
	 * @param X2
	 * @return
	 */
	public static double[][] Pairs( double[][] X1, double[][] X2 ) {
		assert( X1.length == X2.length );
		int N = X1.length;
		int n = X1[0].length;
		int m = X2[0].length;
		
		double[][] result = new double[n][m];
		
		for( int i = 0; i < N; i++ )
		{
			for( int j = 0; j < n; j++)
			{
				for( int k = 0; k < m; k++)
				{
					result[j][k] += ( (X1[i][j] * X2[i][k]) - result[j][k] )/(i+1);
				}
			}
		}
		
		return result;
	}
	
	/**
	 * Extract i-th row of the matrix X
	 * @param X
	 * @param i
	 * @return
	 */
	public static double[] row( double[][] X, int i ) {
		return X[i];
	}
	
	/**
	 * Extract i-th column of the matrix X
	 * @param X
	 * @param i
	 * @return
	 */
	public static double[] col( double[][] X, int i ) {
		double[] y = new double[X.length];
		for( int j = 0; j < X.length; j++ )
			y[j] = X[j][i];
		return y;
	}
	
	/**
	 * Set the rows i to j of the matrix X
	 * @param X
	 * @param i
	 * @param j
	 * @return
	 */
	public static void setRows( double[][] X, int start, int end, double[][] r ) {
		assert( r.length == end-start );
		for( int i = 0; i < end-start; i++ ) X[start+i] = r[i];
	}
	public static void setRow( double[][] X, int i, double[] r ) {
    X[i] = r;
	}

	/**
	 * Set the rows i to j of the matrix X
	 * @param X
	 * @param i
	 * @param j
	 * @return
	 */
	public static void setCols( double[][] X, int start, int end, double[][] c ) {
		assert( c[0].length == end-start );
		for( int i = 0; i < end-start; i++ ) {
      for( int j = 0; j < X.length; j++ ) { 
        X[j][start+i] = c[i][j];
      }
    }
	}
	public static void setCol( double[][] X, int i, double[] c ) {
    double [][] c_ = {c};
    setCols( X, i, i+1, c_ );
	}
	
	/**
	 * Project each columns of X onto a simplex
	 * @param X
	 * @return
	 */
	public static double max( double[] x )
	{
		double max = Double.NEGATIVE_INFINITY;
		for( int i = 0; i < x.length; i++ ) {
			if( x[i] > max ) max = x[i];
		}
		
		return max;
	}	
	public static double max( double[][] x )
	{
		double max = Double.NEGATIVE_INFINITY;
		for( int i = 0; i < x.length; i++ ) {
			for( int j = 0; j < x[i].length; j++ ) {
				if( x[i][j] > max ) max = x[i][j];
			}
		}
		
		return max;
	}	
	
	/**
	 * Find the smallest element value 
	 * @param X
	 * @return
	 */
	public static double min( double[] x )
	{
		double min = Double.POSITIVE_INFINITY;
		for( int i = 0; i < x.length; i++ ) {
			if( x[i] < min ) min = x[i];
		}
		
		return min;
	}	
	public static double min( double[][] x )
	{
		double min = Double.POSITIVE_INFINITY;
		for( int i = 0; i < x.length; i++ ) {
			for( int j = 0; j < x[i].length; j++ ) {
				if( x[i][j] < min ) min = x[i][j];
			}
		}
		
		return min;
	}	

	
	/**
	 * Get maximum value of X
	 * @param X
	 * @return
	 */
	public static int argmax( double[] x )
	{
		int max_i = 0;
		double max = Double.NEGATIVE_INFINITY;
		for( int i = 0; i < x.length; i++ ) {
			if( x[i] > max ) {
				max = x[i]; max_i = i;
			}
		}
		return max_i;
	}	
	
	/**
	 * Get maximum value of X
	 * @param X
	 * @return
	 */
	public static int argmin( double[] x )
	{
		int min_i = 0;
		double min = Double.POSITIVE_INFINITY;
		for( int i = 0; i < x.length; i++ ) {
			if( x[i] > min ) {
				min = x[i]; min_i = i;
			}
		}
		return min_i;
	}	

	public static void printMatrix( double[][] X ) {
		System.out.printf( "rows = %d, cols = %d\n", X.length, X[0].length );
		for(int i = 0; i < X.length; i++ ){
			for(int j = 0; j < X[i].length; j++ ){
				System.out.printf( "%.3f ", X[i][j]);
			}
			System.out.printf( "\n" );
		}
	}

	/**
	 * Find the pairwise distance of the i-th row in X and j-th row in Y
	 * @param X
	 * @param Y
	 * @return
	 */
	public static double[][] cdist(double[][] X, double[][] Y, boolean betweenRows) {
    assert( X.length == Y.length );
    assert( X[0].length == Y[0].length );

		int nRows = X.length;
		int nCols = X[0].length;

    double[][] D;
    if( betweenRows ) {
      D = new double[nRows][nRows];
      for( int i = 0; i < nRows; i++ ) {
        for( int j = 0; j < nRows; j++ ) {
          double dist = 0.0;
          for( int k = 0; k < nCols; k++ )
            dist += (X[i][k] - Y[j][k]) * (X[i][k] - Y[j][k]);
          D[i][j] = Math.sqrt(dist);
        }
      }
    } else {
      D = new double[nCols][nCols];
      for( int i = 0; i < nCols; i++ ) {
        for( int j = 0; j < nCols; j++ ) {
          double dist = 0.0;
          for( int k = 0; k < nRows; k++ )
            dist += (X[k][i] - Y[k][j]) * (X[k][i] - Y[k][j]);
          D[i][j] = Math.sqrt(dist);
        }
      }
    }

    return D;
	}

  /**
   * Align the rows of matrix X so that the rows/columns are matched with the
   * columns of Y
   */
	public static double[][] alignMatrix( double[][] X, double[][] Y, boolean alignRows ) {
    assert( X.length == Y.length );
    assert( X[0].length == Y[0].length );
    int nRows = X.length; 
    int nCols = X[0].length; 

    double[][] X_ = new double[nRows][nCols];

    if( alignRows ) {
      // Populate the weight matrix 
      double[][] W = cdist( Y, X, true );
      // Compute min-weight matching
      int[][] matching = HungarianAlgorithm.findWeightedMatching( W, false );
      // Shuffle rows
      for( int[] match : matching )
        X_[match[0]] = X[ match[1] ];
    } else {
      // Populate the weight matrix 
      double[][] W = cdist( Y, X, false );
      // Compute min-weight matching
      int[][] matching = HungarianAlgorithm.findWeightedMatching( W, false );
      // Shuffle rows
      for( int[] match : matching )
        setCol( X_, match[0], col( X, match[1] ) );
    }

    return X_;
	}

}
