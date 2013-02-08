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
public class SimpleTensor implements Tensor {
  protected SimpleMatrix[] X;

  /**
   * Construct a tensor with three views of data
   * @param X1 - First view; each data point is a row.
   * @param X2
   * @param X3
   */
  public SimpleTensor(SimpleMatrix X1, SimpleMatrix X2, SimpleMatrix X3) {
    // All the Xs should have the same number of rows
    assert( X1.numRows() == X2.numRows() && X2.numRows() == X3.numRows() );

    this.X = new SimpleMatrix[3];
    this.X[0] = X1;
    this.X[1] = X2;
    this.X[2] = X3;
  }

  public int getDim( int axis ) {
    return X[axis].numCols();
  }

  /**
   * Project the tensor onto a matrix by taking an inner product with theta
   * @param axis
   * @param theta
   * @return
   */
  public SimpleMatrix project( int axis, double[] theta )
  {
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
    SimpleMatrix X1 = X[idx1];
    SimpleMatrix X2 = X[idx2];
    SimpleMatrix X3 = X[idx3];
    assert( theta.length == X3.numCols() );

    double[] X1_ = X1.getMatrix().data;
    double[] X2_ = X2.getMatrix().data;
    double[] X3_ = X3.getMatrix().data;

    int nRows = X[0].numRows();

    int n = X1.numCols();
    int m = X2.numCols();
    int l = X3.numCols();


    double[][] Z = new double[n][m]; 
    for( int row = 0; row < nRows; row++ )
    {
      // Compute the dot product
      double prod = 0.0;
      for( int i = 0; i < l; i++ )
        prod += X3_[ row * l + i ] * theta[ i ];

      // Compute the inner product
      for( int i = 0; i < n; i++ ) {
        double x1 = X1_[ row * n + i ];
        for (int j = 0; j < m; j++) {
          double x2 = X2_[ row * m + j ];
          Z[i][j] += (x1 * x2 * prod - Z[i][j])/(row+1);
        }
      }
    }

    return new SimpleMatrix( Z );
  }

  @Override
  public SimpleMatrix project( int axis, SimpleMatrix theta )
  {
    double[] theta_ = theta.getMatrix().data;
    return project( axis, theta_ );
  }

  @Override
  public SimpleMatrix project2(int axis1, int axis2, SimpleMatrix theta1, SimpleMatrix theta2) {
    throw new NoSuchMethodError();
  }

  @Override
  public double project3(SimpleMatrix theta1, SimpleMatrix theta2, SimpleMatrix theta3) {
    throw new NoSuchMethodError();
  }

}

