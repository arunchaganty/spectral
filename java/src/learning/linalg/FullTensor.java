package learning.linalg;

import learning.linalg.Tensor;
import org.ejml.simple.SimpleMatrix;
import org.ejml.data.DenseMatrix64F;
import org.javatuples.Triplet;

/**
  A Tensor constructed with a full DxDxD matrix
 */
public class FullTensor implements Tensor {
  double[][][] X;
  int D1, D2, D3;

  public FullTensor( double[][][] X ) {
    this.X = X;
    D1 = X.length;
    D2 = X[0].length;
    D3 = X[0][0].length;
  }

  @Override
  public SimpleMatrix project(int axis, SimpleMatrix theta) {
    assert( 0 <= axis && axis < 3 );
    // TODO: Handle other axis configurations
    assert( axis == 2 );

    double[] theta_ = MatrixFactory.toVector( theta );
    double[][] Y = new double[ D1 ][ D2 ];
    for( int d = 0; d < D1; d++ ) {
      for( int d_ = 0; d_ < D2; d_++ ) {
        for( int d__ = 0; d__ < D3; d__++ ) {
          Y[d][d_] += theta_[d__] * X[d][d_][d__];
        }
      }
    }

    return new SimpleMatrix( Y );
  }
}
