package learning.utils;

/**
 * General vector operations
 */
public class MatrixOps {
	
	/**
	 * Transpose the matrix x
	 * @param x
	 * @return
	 */
	public static double[][] transpose( double[][] x ) {
		int d = x.length;
		int r = x[0].length;
		
		double[][] result = new double[r][d];
		for( int i = 0; i < d; i++ ) {
			for( int j = 0; j < r; j++ )
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

}
