/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.utils;

import org.ejml.simple.SimpleMatrix;

/**
 * A tensor class that makes it easy to project onto a matrix
 */
public class SimpleTensor implements Tensor {
	protected double[][][] X;
	
	/**
	 * Construct a tensor with three views of data
	 * @param X1 - First view; each data point is a row.
	 * @param X2
	 * @param X3
	 */
	public SimpleTensor(double[][] X1, double[][] X2, double[][] X3) {
		// All the Xs should have the same number of rows
		assert( X1.length == X2.length && X2.length == X3.length );
		
		this.X = new double[3][][];
		this.X[0] = X1;
		this.X[1] = X2;
		this.X[2] = X3;
	}
	
	public SimpleTensor(SimpleMatrix X1, SimpleMatrix X2, SimpleMatrix X3) {
		this.X = new double[3][][];
		this.X[0] = MatrixFactory.fromSimpleMatrix(X1);
		this.X[1] = MatrixFactory.fromSimpleMatrix(X2);
		this.X[2] = MatrixFactory.fromSimpleMatrix(X3);
	}
	
	/**
	 * Project the tensor onto a matrix by taking an inner product with theta
	 * @param axis
	 * @param theta
	 * @return
	 */
	public double[][] project( int axis, double[] theta )
	{
		assert( 0 <= axis && axis < 3 );
		
		// Select the appropriate index
		int idx1, idx2, idx3 = axis;
		switch( axis ){
			case 0:
				idx1 = 1; idx2 = 2;
				break;
			case 1:
				idx1 = 0; idx2 = 2;
				break;
			case 2:
				idx1 = 0; idx2 = 1;
				break;
			default:
				throw new IndexOutOfBoundsException();
		}
		
		double[][] X1 = X[idx1];
		double[][] X2 = X[idx2];
		double[][] X3 = X[idx3];
		assert( theta.length == X3[0].length );
		
		int N = X1.length;
		
		int n = X1[0].length;
		int m = X2[0].length;
		
		double[][] result = new double[n][m]; 
		
		for( int i = 0; i < N; i++ )
		{
			double prod = MatrixOps.dot( X3[i], theta );
			for( int j = 0; j < n; j++ ){
				for (int k = 0; k < m; k++) {
					result[j][k] += (X1[i][j] * X2[i][k] * prod - result[j][k])/(i+1);
				}
			}
		}
		
		return result;
	}
	
	@Override
	public SimpleMatrix project( int axis, SimpleMatrix theta )
	{
		double[] theta_ = theta.getMatrix().data;
		double[][] result = project( axis, theta_ );
		return new SimpleMatrix(result);
	}
	
}
