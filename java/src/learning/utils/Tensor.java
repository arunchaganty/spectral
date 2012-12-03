package learning.utils;

import org.ejml.simple.SimpleMatrix;

public interface Tensor {
	/**
	 * Project the tensor onto a matrix by taking an inner product with theta
	 * @param axis
	 * @param theta
	 * @return
	 */
	public SimpleMatrix project( int axis, SimpleMatrix theta );
}