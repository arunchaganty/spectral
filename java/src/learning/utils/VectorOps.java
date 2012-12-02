package learning.utils;

/**
 * General vector operations
 */
public class VectorOps {
	
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
	 * Normalize the entries of x to sum to 1
	 * Note: Changes x in place
	 * @param x
	 */
	public static void normalize( double[] x ) {
		double sum = sum(x);
		for( int i = 0; i < x.length; i++ )
			x[i] /= sum;
	}

}
