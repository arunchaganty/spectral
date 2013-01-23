/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.linalg;

import learning.linalg.Tensor;
import org.ejml.simple.SimpleMatrix;
import org.ejml.data.DenseMatrix64F;

/**
 * A tensor class that makes it easy to project onto a matrix
 */
public class ExactTensor implements Tensor {
  protected SimpleMatrix weights;
  protected SimpleMatrix[] M;

  /**
   * Construct a tensor with three views of data
   * @param X1 - First view; each data point is a row.
   * @param X2
   * @param X3
   */
  public ExactTensor(SimpleMatrix weights, SimpleMatrix M1, SimpleMatrix M2, SimpleMatrix M3) {
    // All the Xs should have the same number of rows
    assert( M1.numRows() == M2.numRows() && M2.numRows() == M3.numRows() );
    assert( M1.numCols() == M2.numCols() && M2.numCols() == M3.numCols() );
    assert( M1.numCols() == weights.numCols() );

    this.weights = weights;
    this.M = new SimpleMatrix[3];
    this.M[0] = M1;
    this.M[1] = M2;
    this.M[2] = M3;
  }

  public int getDim( int axis ) {
    return M[axis].numRows();
  }

  /**
   * Project the tensor onto a matrix by taking an inner product with theta
   * @param axis
   * @param theta
   * @return
   */
  public SimpleMatrix project( int axis, double[] theta ) {
    assert( 0 <= axis && axis < 3 );

    // Align the axes
    int idx1, idx2, idx3 = axis;
    switch( axis ){
      case 0: idx1 = 1; idx2 = 2; break;
      case 1: idx1 = 0; idx2 = 2; break;
      case 2: idx1 = 0; idx2 = 1; break;
      default:
              throw new IndexOutOfBoundsException();
    }

    // Canonicalise the names
    SimpleMatrix M1 = M[idx1];
    SimpleMatrix M2 = M[idx2];
    SimpleMatrix M3 = M[idx3];
    assert( theta.length == M3.numRows() );

    
    double[] weights_ = MatrixFactory.toVector( weights );
    double[][] M1_ = MatrixFactory.toArray( M1 );
    double[][] M2_ = MatrixFactory.toArray( M2 );
    double[][] M3_ = MatrixFactory.toArray( M3 );

    int n = M1.numRows();
    int m = M2.numRows();
    int l = M3.numRows();

    int K = M1.numCols();

    double[][] Z = new double[n][m]; 
    for( int k = 0; k < K; k++ ) {
      // Compute the dot product
      double prod = 0.0;
      for( int i = 0; i < l; i++ )
        prod += M3_[i][k] * theta[ i ];

      // Compute the inner product
      for( int i = 0; i < n; i++ ) {
        double x1 = M1_[i][k];
        for (int j = 0; j < m; j++) {
          double x2 = M2_[j][k];
          Z[i][j] += x1 * x2 * prod * weights_[k];
        }
      }
    }

    return new SimpleMatrix( Z );
  }

  @Override
  public SimpleMatrix project( int axis, SimpleMatrix theta ) {
    double[] theta_ = theta.getMatrix().data;
    return project( axis, theta_ );
  }

}

