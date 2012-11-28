package learning.utils;

/**
 * General vector operations
 */
public class VectorOps {
	
	public static double dot( double[] x, double[] y ) {
		assert( x.length == y.length );
		double prod = 0.0;
		for( int i = 0; i < x.length; i++ )
			prod += x[i] * y[i];
		
		return prod;
	}

}
