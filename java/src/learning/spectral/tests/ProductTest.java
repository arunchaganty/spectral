package learning.spectral.tests;

/**
 * This is a profiling test for computing moments
 */
public class ProductTest {
	
	
	public static double[][] Pairs(double[][] X1, double[][] X2 ) {
		assert( X1.length == X2.length );
		assert( X1[0].length == X2[0].length );
		
		int n = X1.length;
		int d = X1[0].length;
		
		double[][] P12 = new double[d][d];
		for( int i = 0; i < n; i++ ) {
			double[] x1 = X1[i];
			double[] x2 = X2[i];
			// Add into P12
			for( int j = 0; j < d; j++ ) {
				for( int k = 0; k < d; k++ ) {
					P12[j][k] += (x1[j] * x2[k] - P12[j][k])/(i+1);
				}
			}
		}
		
		return P12;
	}
			
	public static void main(String[] args) {
		for( int log_d = 3; log_d <= 3; log_d++ ) {
			for(int log_n = 3; log_n <= 4; log_n++ ) {
				int n = (int) Math.pow( 10, log_n );
				int d = (int) Math.pow( 10, log_d );
				
				// Values really don't matter
				double[][] X1 = new double[n][d];
				double[][] X2 = new double[n][d];
				
				long start = System.nanoTime();
				Pairs( X1, X2 );
				long stop = System.nanoTime();
				
				System.out.printf("%d %d: %f\n", n, d, (stop - start)/Math.pow(10,9));
			}
		}
		
		
	}

}
