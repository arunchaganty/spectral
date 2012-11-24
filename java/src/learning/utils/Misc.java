package learning.utils;

import java.util.Comparator;

/**
 * Miscellaneous routines 
 */
public class Misc {	
	
	/**
	 * Comparator to sort a list based on keys in another list 
	 */
	public static class IndexedComparator implements Comparator<Integer> {
		double[] keys;
		
		public IndexedComparator(double[] keys) {
			this.keys = keys;
		}
		
		@Override
		public int compare(Integer o1, Integer o2) {
			if( keys[o2] - keys[o1] < 0 )
				return -1;
			else if( keys[o2] - keys[o1] > 0 )
				return 1;
			else
				return 0;
		}
	}
	
	/**
	 * Renormalises x to a probability distribution in-place
	 * @param x
	 * @return
	 */
	public static double[] renormalize( double[] x ) {
		double total = 0.0;
		for(int i = 0; i < x.length; i++) {
			assert( x[i] >= 0.0 );
			total += x[i];
		}
		for(int i = 0; i < x.length; i++) x[i] /= total;
		return x;
	}
	
	/**
	 * Convert an boxed integer array into an unboxed integer array
	 * @param x
	 * @return
	 */
	public static int[] toPrimitive( Integer[] x ) {
		final int[] y = new int[x.length];
		for(int i = 0; i < x.length; i++ )
			y[i] = x[i].intValue();
		return y;
	}
	
	/**
	 * Convert an boxed double array into an unboxed double array
	 * @param x
	 * @return
	 */
	public static double[] toPrimitive( Double[] x ) {
		final double[] y = new double[x.length];
		for(int i = 0; i < x.length; i++ )
			y[i] = x[i].doubleValue();
		return y;
	}	
	
	public static class NotImplementedException extends Exception {
		private static final long serialVersionUID = 4046853531401489206L;
		
		public NotImplementedException() {
			super();
		}
		public NotImplementedException(String message) {
			super(message);
		}
	}
	
}
