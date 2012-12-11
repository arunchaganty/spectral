/**
 * learning.linalg
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.linalg;

import learning.linalg.MatrixOps;
import learning.linalg.RandomFactory;
import learning.exceptions.NumericalException;

import org.ejml.simple.SimpleMatrix;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

/**
 * 
 */
public class RandomFactoryTest {

  @Before
  public void setUp() {
  }

  @Test
  public void randn() {
    SimpleMatrix X1 = RandomFactory.randn( 3, 4 );
    double max = MatrixOps.max( X1 );
    double min = MatrixOps.min( X1 );

    Assert.assertTrue( X1.numRows() == 3 );
    Assert.assertTrue( X1.numCols() == 4 );
    // With very high probability this will be true (Pr ~= 1 - 1/2^12)
    Assert.assertTrue( max > 0 );
    Assert.assertTrue( min < 0 );
  }

  @Test
  public void orthogonal() {
    SimpleMatrix X1 = RandomFactory.orthogonal( 3 );

    // Verify orthogonality property
    Assert.assertTrue( MatrixOps.allclose( X1.mult( X1.transpose() ), MatrixFactory.eye( 3 ) ) );
  }

  @Test
  public void multinomial() {
    double[] pi_ = {0.1, 0.7, 0.2};
    SimpleMatrix pi = MatrixFactory.fromVector( pi_ );

    for( int i = 0; i < 10; i++ ) {
      int choice = RandomFactory.multinomial( pi );
      Assert.assertTrue( 0 <= choice && choice <= 2 );
    }

    SimpleMatrix counts = RandomFactory.multinomial( pi, 100 );

    // True with very high probability
    Assert.assertTrue( counts.get(1) > counts.get(0) && counts.get(1) > counts.get(2) );
  }

  @Test
  public void multivariateGaussian() {
    double[][] mean_ = {{ -1.0, 1.0 }};
    double[][] cov_ = {
      { 1.0, 0.3 },
      { 0.3, 1.0 }};

    SimpleMatrix mean = new SimpleMatrix( mean_ );
    SimpleMatrix cov = new SimpleMatrix( cov_ );

    SimpleMatrix X = RandomFactory.multivariateGaussian( mean, cov, 100 );
    
    Assert.assertTrue( X.numRows() == 100 );
    Assert.assertTrue( X.numCols() == 2 );
  }

}
