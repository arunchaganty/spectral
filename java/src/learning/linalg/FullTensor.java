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

  public FullTensor clone() {
    double[][][] Y = new double[D1][D2][D3];
    for(int d1 = 0; d1 < D1; d1++) {
      for(int d2 = 0; d2 < D2; d2++) {
        System.arraycopy(X[d1][d2], 0, Y[d1][d2], 0, D3);
      }
    }
    return new FullTensor(Y);
  }

  public static FullTensor fromUnitVector( double[] x ) {
    int D = x.length;
    double[][][] X = new double[D][D][D];

    for(int d1 = 0; d1 < D; d1++) {
      for(int d2 = 0; d2 < D; d2++) {
        for(int d3 = 0; d3 < D; d3++) {
          X[d1][d2][d3] = x[d1] * x[d2] * x[d3];
        }
      }
    }

    return new FullTensor(X);
  }
  public static FullTensor fromUnitVector( SimpleMatrix x ) {
    return fromUnitVector(MatrixFactory.toVector(x));
  }

  public static FullTensor fromDecomposition( double[] scale, double[][] X ) {
    int K = X.length;
    int D = X[0].length;
    assert( K == scale.length );

    double[][][] Y = new double[D][D][D];

    for(int d1 = 0; d1 < D; d1++) {
      for(int d2 = 0; d2 < D; d2++) {
        for(int d3 = 0; d3 < D; d3++) {
          Y[d1][d2][d3] = 0;
          for( int k = 0; k < K; k++ )
            Y[d1][d2][d3] += scale[k] * X[k][d1] * X[k][d2] * X[k][d3];
        }
      }
    }

    return new FullTensor(Y);
  }

  /**
   * @param scale
   * @param X - assumes X is in D x K layout
   * @return
   */
  public static FullTensor fromDecomposition( SimpleMatrix scale, SimpleMatrix X ){
    return fromDecomposition(
            MatrixFactory.toVector(scale),
            MatrixFactory.toArray(X.transpose()));
  }

  public static FullTensor fromDecomposition( double[] scale, double[][] X1, double[][] X2, double[][] X3 ) {
    int K = X1.length;
    assert( scale.length == K && X2.length == K  && X3.length == K );

    int D1 = X1[0].length;
    int D2 = X2[0].length;
    int D3 = X3[0].length;

    double[][][] Y = new double[D1][D2][D3];

    for(int d1 = 0; d1 < D1; d1++) {
      for(int d2 = 0; d2 < D2; d2++) {
        for(int d3 = 0; d3 < D3; d3++) {
          Y[d1][d2][d3] = 0;
          for( int k = 0; k < K; k++ )
            Y[d1][d2][d3] += scale[k] * X1[k][d1] * X2[k][d2] * X3[k][d3];
        }
      }
    }

    return new FullTensor(Y);
  }

  /**
   * Construct a tensor from M1, M2, M3, where
   * T = \sum_k w_k M1_^(k) \otimes M2_^(k) \otimes M3_^(k)
   * @param weights
   * @param X1 - Assumed to be in D x K layout.
   * @param X2 - Assumed to be in D x K layout.
   * @param X3 - Assumed to be in D x K layout.
   * @return
   */
  public static FullTensor fromDecomposition( SimpleMatrix weights, SimpleMatrix X1, SimpleMatrix X2, SimpleMatrix X3 ) {
    return fromDecomposition( MatrixFactory.toVector(weights),
            MatrixFactory.toArray(X1.transpose()),
            MatrixFactory.toArray(X2.transpose()),
            MatrixFactory.toArray(X3.transpose()));
  }

  public int getDim( int axis ) {
    switch( axis ) {
      case 0: return D1;
      case 1: return D2;
      case 2: return D3;
      default:
        throw new IllegalArgumentException();
    }
  }

  public void set(int i, int j, int k, double x) {
    X[i][j][k] = x;
  }
  public double get(int i, int j, int k) {
    return X[i][j][k];
  }
  public FullTensor plus(double scale, FullTensor B) {
    double Y[][][] = X.clone();
    for(int d1 = 0; d1 < D1; d1++) {
      for(int d2 = 0; d2 < D2; d2++) {
        for(int d3 = 0; d3 < D3; d3++) {
          Y[d1][d2][d3] += scale * B.X[d1][d2][d3];
        }
      }
    }
    return new FullTensor(Y);
  }
  public FullTensor plus(FullTensor B) {
    return plus(1.0, B);
  }
  public FullTensor minus(double scale, FullTensor B) {
    return plus(-scale, B);
  }
  public FullTensor minus(FullTensor B) {
    return plus(-1.0, B);
  }
  public FullTensor scale(double scale) {
    double Y[][][] = X.clone();
    for(int d1 = 0; d1 < D1; d1++) {
      for(int d2 = 0; d2 < D2; d2++) {
        for(int d3 = 0; d3 < D3; d3++) {
          Y[d1][d2][d3] *= scale;
        }
      }
    }
    return new FullTensor(Y);
  }

  /**
   * Rotate the axis of the tensor by the matrix M,
   * i.e., T_{ijk} = T_{ijp} X_{pk}
   * @param axis
   * @param M
   * @return
   */
  public FullTensor rotate(int axis, SimpleMatrix M) {
    assert( 0 <= axis && axis <= 2 );

    int L1, L2, L3;
    switch(axis) {
      case 0: L1 = M.numCols(); L2 = D2; L3 = D3; break;
      case 1: L1 = D1; L2 = M.numCols(); L3 = D3; break;
      case 2: L1 = D1; L2 = D2; L3 = M.numCols(); break;
      default:
        throw new NoSuchMethodError("invalid axis");
    }

    double[][][] Y = new double[L1][L2][L3];
    for(int l1 = 0; l1 < L1; l1++) {
      for(int l2 = 0; l2 < L2; l2++) {
        for(int l3 = 0; l3 < L3; l3++) {
          Y[l1][l2][l3] = 0;
          for(int d = 0; d < getDim(axis); d++)
            switch(axis) {
              case 0:
                Y[l1][l2][l3] += X[d][l2][l3] * M.get(d, l1); break;
              case 1:
                Y[l1][l2][l3] += X[l1][d][l3] * M.get(d, l2); break;
              case 2:
                Y[l1][l2][l3] += X[l1][l2][d] * M.get(d, l3); break;
              default:
                throw new NoSuchMethodError("Invalid axis");
            }
        }
      }
    }
    return new FullTensor(Y);
  }

  /**
   * Rotate the tensor about each axis using M1, M2 and M3.
   * Set any to null if you do not want to rotate along that axis (i.e. rotate with identity).
   * @param M1
   * @param M2
   * @param M3
   * @return
   */
  public FullTensor rotate(SimpleMatrix M1, SimpleMatrix M2, SimpleMatrix M3) {
    FullTensor Y = this.rotate(0, M1);
    Y = Y.rotate(1, M2);
    Y = Y.rotate(2, M3);

    return Y;
  }

    // Buggy method?
  @Deprecated
  public FullTensor rotateSlow(SimpleMatrix M1, SimpleMatrix M2, SimpleMatrix M3) {
    int L1 = M1.numCols();
    int L2 = M2.numCols();
    int L3 = M3.numCols();
    double[][][] Y = new double[L1][L2][L3];
    for(int l1 = 0; l1 < L1; l1++) {
      for(int l2 = 0; l2 < L2; l2++) {
        for(int l3 = 0; l3 < L3; l3++) {
          Y[l1][l2][l3] = 0;
          for(int d1 = 0; d1 < D1; d1++)
            for(int d2 = 0; d2 < D2; d2++)
              for(int d3 = 0; d3 < D3; d3++)
                Y[l1][l2][l3] += X[d1][d2][d3] * M1.get(d1, l1) * M2.get(d2, l2) * M3.get(d3, l3);
        }
      }
    }
    return new FullTensor(Y);
  }

  @Override
  public SimpleMatrix project(int axis, SimpleMatrix theta) {
    int L1, L2;
    switch( axis ) {
      case 0: L1 = D2; L2 = D3; break;
      case 1: L1 = D1; L2 = D3; break;
      case 2: L1 = D1; L2 = D2; break;
      default: throw new NoSuchMethodError( "Invalid axis" );
    }

    double[] theta_ = MatrixFactory.toVector( theta );
    double[][] Y = new double[ L1 ][ L2 ];
    for( int d1 = 0; d1 < D1; d1++ ) {
      for( int d2 = 0; d2 < D2; d2++ ) {
        for( int d3 = 0; d3 < D3; d3++ ) {
          switch (axis) {
            case 0:
              Y[d2][d3] += theta_[d1] * X[d1][d2][d3]; break;
            case 1:
              Y[d1][d3] += theta_[d2] * X[d1][d2][d3]; break;
            case 2:
              Y[d1][d2] += theta_[d3] * X[d1][d2][d3]; break;
          }
        }
      }
    }

    return new SimpleMatrix( Y );
  }

  public SimpleMatrix project2( int axis1, int axis2, SimpleMatrix theta1, SimpleMatrix theta2 ) {
    assert( 0 <= axis1 && axis1 < 3 );
    assert( 0 <= axis2 && axis2 < 3 );

    // The other axis
    int axis = 0 + 1 + 2 - axis1 - axis2;
    int L;
    switch(axis) {
      case 0: L = D1; break;
      case 1: L = D2; break;
      case 2: L = D3; break;
      default: throw new NoSuchMethodError();
    }

    double[] theta1_ = MatrixFactory.toVector( theta1 );
    double[] theta2_ = MatrixFactory.toVector( theta2 );
    double[] y = new double[ L ];
    for( int d1 = 0; d1 < D1; d1++ ) {
      for( int d2 = 0; d2 < D2; d2++ ) {
        for( int d3 = 0; d3 < D3; d3++ ) {
          double scalar = 1.0;
          switch(axis1) {
            case 0: scalar *= theta1_[d1]; break;
            case 1: scalar *= theta1_[d2]; break;
            case 2: scalar *= theta1_[d3]; break;
          }
          switch(axis2) {
            case 0: scalar *= theta2_[d1]; break;
            case 1: scalar *= theta2_[d2]; break;
            case 2: scalar *= theta2_[d3]; break;
          }
          switch(axis) {
            case 0: y[d1] += scalar * X[d1][d2][d3]; break;
            case 1: y[d2] += scalar * X[d1][d2][d3]; break;
            case 2: y[d3] += scalar * X[d1][d2][d3]; break;
          }
        }
      }
    }
    return MatrixFactory.fromVector(y);
  }

  public double project3(SimpleMatrix theta1, SimpleMatrix theta2, SimpleMatrix theta3) {
    double[] theta1_ = MatrixFactory.toVector( theta1 );
    double[] theta2_ = MatrixFactory.toVector( theta2 );
    double[] theta3_ = MatrixFactory.toVector( theta3 );
    double y = 0.0;
    for( int d1 = 0; d1 < D1; d1++ ) {
      for( int d2 = 0; d2 < D2; d2++ ) {
        for( int d3 = 0; d3 < D3; d3++ ) {
          y += theta1_[d1] * theta2_[d2] * theta3_[d3] * X[d1][d2][d3];
        }
      }
    }
    return y;
  }

}