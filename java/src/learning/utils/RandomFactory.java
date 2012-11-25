package learning.utils;

import java.util.Random;

import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.factory.QRDecomposition;
import org.ejml.simple.SimpleMatrix;

/**
 * A set of functions to generate random variables
 */
public class RandomFactory {
	private static Random rand = new Random();
	
	/**
	 * Generate a random matrix with standard normal entries.
	 * @param d
	 * @return
	 */
	public static SimpleMatrix randn(int m, int n) {
		SimpleMatrix X = MatrixFactory.zeros(m,n);
		for( int i = 0; i < m; i++)
			for( int j = 0; j < n; j++)
				X.set( i, j, rand.nextGaussian() );
		
		return X;
	}
	
	/**
	 * Generate a single random variable
	 * @param sigma - noise
	 * @return
	 */
	public static double randn(double sigma) {
		return rand.nextGaussian() * sigma;
	}

	/**
	 * Generate a random orthogonal 'd' dimensional matrix, using the
     * the technique described in: Francesco Mezzadri, "How to generate 
     * random matrices from the classical compact groups" 
	 * @param d
	 * @return
	 */
	public static SimpleMatrix orthogonal(int d) {
		SimpleMatrix Z = randn(d,d);
		QRDecomposition<DenseMatrix64F> Z_QR = DecompositionFactory.qr(Z.numRows(), Z.numCols());
		Z_QR.decompose(Z.getMatrix());
		SimpleMatrix Q = SimpleMatrix.wrap( Z_QR.getQ(null, true) );
		SimpleMatrix R = SimpleMatrix.wrap( Z_QR.getR(null, true) ); 
		SimpleMatrix D = MatrixFactory.diag(R);
		for( int i = 0; i < d; i++)
			D.set(i, D.get(i)/Math.abs(D.get(i)));
		return Q.mult(MatrixFactory.diag(D));
	}

	/**
	 * Draw an element from a multinomial distribution with weights given in matrix.
	 * @param pi
	 * @return
	 */
	public static int multinomial(SimpleMatrix pi) {
		assert( pi.numCols() == 1 );
		
		double x = rand.nextDouble();
		for( int i = 0; i < pi.numRows(); i++ )
		{
			if( x <= pi.get(i) )
				return i;
			else
				x -= pi.get(i);
		}
		
		// The remaining probability is assigned to the last element in the sequence.
		return pi.numRows()-1;
	}

	/**
	 * Draw an element from a multinomial distribution with weights given in matrix.
	 * @param pi
	 * @return
	 */
	public static int multinomial(double[] pi) {
		double x = rand.nextDouble();
		for( int i = 0; i < pi.length; i++ )
		{
			if( x <= pi[i] )
				return i;
			else
				x -= pi[i];
		}
		
		// The remaining probability is assigned to the last element in the sequence.
		return pi.length-1;
	}
	
	/**
	 * Draw many elements from a multinomial distribution with weights given in matrix.
	 * @param n - Number of draws
	 * @param pi - Parameters
	 * @return - Vector with count of number of times a value was drawn
	 */
	public static SimpleMatrix multinomial(int n, SimpleMatrix pi) {
		assert( pi.numCols() == 1 );
		double[] cnt = new double[pi.numRows()];
		
		for( int i = 0; i < n; i++)
			cnt[ multinomial(pi) ] += 1;
		
		double[][] cnt_ = {cnt};
		return new SimpleMatrix( cnt_ );
	}

}
