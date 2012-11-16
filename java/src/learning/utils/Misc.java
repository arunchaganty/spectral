package learning.utils;

/**
 * Miscellaneous routines 
 */
public class Misc {	
	
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
}
