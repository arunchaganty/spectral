package learning.models.transforms;

import java.io.Serializable;

/**
 * Represents a generic non-linear transform of the data
 */
public class NonLinearity implements Serializable {
  public int getLinearDimension( int dimension  ) {return dimension;}
  public double[] getLinearEmbedding( final double[] x ) {return x;}
  public double[][] getLinearEmbedding( final double[][] X ) {
    int N = X.length;
    double[][] Y = new double[ N ][];

    for( int n = 0 ; n < N; n ++ ) {
      Y[n] = getLinearEmbedding( X[n] );
    }

    return Y;
  }
}
