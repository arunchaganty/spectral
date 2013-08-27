/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.linalg;

import learning.linalg.MatrixOps;
import learning.exceptions.NumericalException;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.Triplet;
import org.junit.Assert;
import org.junit.Test;
import org.junit.Before;

/**
 * 
 */
public class MatrixOpsTest {

  SimpleMatrix X1;
  SimpleMatrix X2;
  SimpleMatrix X3;
  SimpleMatrix X4;

  FullTensor T; // Random tensor
  FullTensor symT; // Random symmetric tensor

  static double EPS_ZERO = 1e-7;
  static double EPS_CLOSE = 1e-4;

  @Before
  public void setUp() {
    double[][] X1_ = {
      { 1.74702881, -1.11241135,  0.18658948},
      {-0.05982937,  0.22105371, -1.49155831},
      { 0.95235621, -0.81554199, -0.33258448}};

    double[][] X2_ = {
       {-1.28215659,  1.1546932 , -2.52303139},
       { 0.28178498, -1.36063266,  0.14748404},
       {-0.84325768, -1.57915379,  1.08970879}};

    double[][] X3_ = {
      { 1.32421528, -0.93429939, -2.14364265},
      { 0.32112581,  0.59689873, -1.05131256},
      { 0.61709072, -0.26502982, -0.83136217}};

    double[][] X4_ = {
      {-1.28215659,  1.1546932 , -2.52303139},
      { 0.28178498, -1.36063266,  0.14748404},
      {-0.84325768, -1.57915379,  1.08970879},
      { 1.32421528, -0.93429939, -2.14364265},
      { 0.32112581,  0.59689873, -1.05131256},
      { 0.61709072, -0.26502982, -0.83136217}};

    double [][][] T_ = {
      { { 0.602906, 0.621318, 0.259487 },
        { 0.980476, 0.096308, 0.763573 },
        { 0.642083, 0.543104, 0.142096 } },
      { { 0.257114, 0.148483, 0.788865 },
          { 0.226163, 0.372460, 0.065712 },
          { 0.853019, 0.343047, 0.794350 } },
      { { 0.091767, 0.159335, 0.594195 },
        { 0.483753, 0.914964, 0.282069 },
        { 0.973374, 0.956173, 0.545773 } }
    };

    double [][][] symT_ = {
      {
        { 8.0386e-01, 5.3795e-03, 2.7792e-01 },
        { 5.3795e-03, 3.6000e-05, 1.8599e-03 },
        { 2.7792e-01, 1.8599e-03, 9.6089e-02 } },
      {
        { 5.3795e-03, 3.6000e-05, 1.8599e-03 },
        { 3.6000e-05, 2.4091e-07, 1.2446e-05 },
        { 1.8599e-03, 1.2446e-05, 6.4303e-04 } },
      {
        { 2.7792e-01, 1.8599e-03, 9.6089e-02 },
        { 1.8599e-03, 1.2446e-05, 6.4303e-04 },
        { 9.6089e-02, 6.4303e-04, 3.3221e-02 } }
    };

    X1 = new SimpleMatrix(X1_);
    X2 = new SimpleMatrix(X2_);
    X3 = new SimpleMatrix(X3_);
    X4 = new SimpleMatrix(X4_);

    T = new FullTensor(T_);
    symT = new FullTensor(symT_);
  }

  @Test
  public void allclose() {
    double[][] P12_1_ = {
      {-1.01996841,  0.19825706, -1.12628049},
      { 0.72542907, -0.0991335 ,  0.65018246},
      {-0.12636041,  0.92337286, -0.35105746}};
    double[][] P12_2_ = {
      {-1.01996,  0.19825, -1.12628},
      { 0.72542, -0.09913,  0.65018},
      {-0.12636,  0.92337, -0.35105}};
    double[][] P12_3_ = {
      {-1.01,  0.19, -1.12},
      { 0.72, -0.09,  0.65},
      {-0.12,  0.92, -0.35}};
    SimpleMatrix P12_1 = new SimpleMatrix( P12_1_ );
    SimpleMatrix P12_2 = new SimpleMatrix( P12_2_ );
    SimpleMatrix P12_3 = new SimpleMatrix( P12_3_ );

    Assert.assertTrue( MatrixOps.allclose( P12_1, P12_2 ) );
    Assert.assertFalse( MatrixOps.allclose( P12_1, P12_3 ) );
    Assert.assertTrue( MatrixOps.allclose( P12_1, P12_3, 1e-1 ) );
  }

  @Test
  public void abs() {
    double[][] X1a_ = {
      { 1.74702881, 1.11241135, 0.18658948},
      { 0.05982937, 0.22105371, 1.49155831},
      { 0.95235621, 0.81554199, 0.33258448}};
    SimpleMatrix X1a = new SimpleMatrix( X1a_ );
    SimpleMatrix X1b = MatrixOps.abs( X1 );

    Assert.assertTrue( MatrixOps.allclose( X1a, X1b ) );
  }

  @Test
  public void min() {
    double min = MatrixOps.min( X1 );
    Assert.assertTrue( min == -1.49155831 );
  }

  @Test
  public void max() {
    double max = MatrixOps.max( X1 );
    Assert.assertTrue( max == 1.74702881 );
  }

  @Test
  public void Pairs() {
    double[][] P12_ = {
      {-1.01996841,  0.19825706, -1.12628049},
      { 0.72542907, -0.0991335 ,  0.65018246},
      {-0.12636041,  0.92337286, -0.35105746}};
    SimpleMatrix P12 = new SimpleMatrix( P12_ );

    SimpleMatrix P12a = MatrixOps.Pairs( X1, X2 );
    Assert.assertTrue( MatrixOps.allclose( P12a, P12 ) );
  }

  @Test
  public void Triples() {
    double[] theta1 = { 0.63137104,  0.59910771,  0.02021931};
    double[] theta2 = { 0.82945477,  0.23272397,  0.60360884};
    
    double[][] P123_1_ = {
      {-0.23427932,  0.0640017 , -0.26986257},
      { 0.17102172, -0.06192624,  0.16042424},
      {-0.07409908,  0.41889465, -0.10194631}};

    double[][] P123_2_ = {
      { 0.32346702, -0.25802949,  0.58958525},
      {-0.21294266,  0.17764126, -0.37355439},
      { 0.06023118, -0.19382615,  0.08786012}};

    SimpleMatrix P123_1 = new SimpleMatrix( P123_1_ );
    SimpleMatrix P123_2 = new SimpleMatrix( P123_2_ );

    FullTensor P123_ = MatrixOps.Triples( X1, X2, X3 );
    SimpleMatrix P123_1a = P123_.project( 2, MatrixFactory.fromVector(theta1) );
    SimpleMatrix P123_2a = P123_.project( 2, MatrixFactory.fromVector(theta2) );

    Assert.assertTrue( MatrixOps.allclose( P123_1, P123_1a ) );
    Assert.assertTrue( MatrixOps.allclose( P123_2, P123_2a ) );
  }

  @Test
	public void col() {
    double[][] X11_ = { {-1.11241135}, {0.22105371}, {-0.81554199},};
    SimpleMatrix X11a = new SimpleMatrix( X11_ );
    SimpleMatrix X11 = MatrixOps.col( X1, 1 );

    Assert.assertTrue( MatrixOps.allclose( X11, X11a ) );
  }

  @Test
	public void row() {
    double[][] X11_ = { {-0.05982937,  0.22105371, -1.49155831},};
    SimpleMatrix X11a = new SimpleMatrix( X11_ );
    SimpleMatrix X11 = MatrixOps.row( X1, 1 );

    Assert.assertTrue( MatrixOps.allclose( X11, X11a ) );
  }

  @Test
	public void setCol() {
    double[][] theta = {
      { 0.63137104,},
      { 0.59910771,},
      { 0.02021931,}};

    double[][] X1a_ = {
      { 1.74702881, 0.63137104,  0.18658948},
      {-0.05982937, 0.59910771, -1.49155831},
      { 0.95235621, 0.02021931, -0.33258448}};

    SimpleMatrix X1a = new SimpleMatrix( X1a_ );

    SimpleMatrix X1b = new SimpleMatrix(X1);
    MatrixOps.setCol( X1b, 1, new SimpleMatrix(theta) );
    Assert.assertTrue( MatrixOps.allclose( X1a, X1b ) );

    X1b = new SimpleMatrix(X1);
    MatrixOps.setCol( X1b, 1, (new SimpleMatrix(theta)).transpose() );
    Assert.assertTrue( MatrixOps.allclose( X1a, X1b ) );
  }

  @Test
	public void setRow() {
    double[][] theta = {{ 0.63137104,  0.59910771,  0.02021931}};

    double[][] X1a_ = {
      { 1.74702881, -1.11241135,  0.18658948},
      { 0.63137104,  0.59910771,  0.02021931},
      { 0.95235621, -0.81554199, -0.33258448}};

    SimpleMatrix X1a = new SimpleMatrix( X1a_ );
    
    SimpleMatrix X1b = new SimpleMatrix(X1);
    MatrixOps.setRow( X1b, 1, new SimpleMatrix(theta) );
    Assert.assertTrue( MatrixOps.allclose( X1a, X1b ) );

    X1b = new SimpleMatrix(X1);
    MatrixOps.setRow( X1b, 1, (new SimpleMatrix(theta)).transpose() );
    Assert.assertTrue( MatrixOps.allclose( X1a, X1b ) );
  }
	
  @Test
	public void setCols() {
    double[][] theta = {
      { 0.63137104,  0.82945477},
      { 0.59910771,  0.23272397},
      { 0.02021931,  0.60360884}};

    double[][] X1a_ = {
      { 1.74702881, 0.63137104,  0.82945477},
      {-0.05982937, 0.59910771,  0.23272397},
      { 0.95235621, 0.02021931,  0.60360884}};

    SimpleMatrix X1a = new SimpleMatrix( X1a_ );
    SimpleMatrix X1b = new SimpleMatrix(X1);
    MatrixOps.setCols( X1b, 1, 3, new SimpleMatrix(theta) );
    Assert.assertTrue( MatrixOps.allclose( X1a, X1b ) );
  }

  @Test
	public void setRows() {
    double[][] theta = {
      { 0.63137104,  0.59910771,  0.02021931},
      { 0.82945477,  0.23272397,  0.60360884}};

    double[][] X1a_ = {
      { 1.74702881, -1.11241135,  0.18658948},
      { 0.63137104,  0.59910771,  0.02021931},
      { 0.82945477,  0.23272397,  0.60360884}};

    SimpleMatrix X1a = new SimpleMatrix( X1a_ );
    SimpleMatrix X1b = new SimpleMatrix(X1);
    MatrixOps.setRows( X1b, 1, 3, new SimpleMatrix(theta) );
    Assert.assertTrue( MatrixOps.allclose( X1a, X1b ) );
  }

  @Test
	public void sum() {
    double[] x = {-0.05982937,  0.22105371, -1.49155831};
    double sum = -1.3303339757731367;
    double sum_ = MatrixOps.sum( x );
    Assert.assertTrue( Math.abs( sum - sum_ ) < EPS_ZERO );
  }

  @Test
	public void rowSum() {
    double sum = -1.3303339757731367;
    double sum_ = MatrixOps.rowSum( X1, 1 );
    Assert.assertTrue( Math.abs( sum - sum_ ) < EPS_ZERO );
  }
	
  @Test
	public void columnSum() {
    double sum = -1.706899636542059;
    double sum_ = MatrixOps.columnSum( X1, 1 );
    Assert.assertTrue( Math.abs( sum - sum_ ) < EPS_ZERO );
  }

  @Test
	public void normalize() {
    double[] x = {-0.05982937,  0.22105371, -1.49155831};
    double[] y = { 0.0449732 , -0.16616407,  1.12119088};
    MatrixOps.normalize( x );
    Assert.assertTrue( MatrixOps.allclose( x, y ) );
  }
  @Test

	public void norm() {
    double[] x = {-0.05982937,  0.22105371, -1.49155831};
    double norm = 1.5090362780097162;
    double norm_ = MatrixOps.norm( x );
    Assert.assertTrue( Math.abs( norm - norm_ ) < EPS_ZERO );
  }

  @Test
	public void makeUnitVector() {
    double[] x = {-0.05982937,  0.22105371, -1.49155831};
    double[] y = {-0.0396474 ,  0.14648668, -0.98841779};
    MatrixOps.makeUnitVector( x );
    Assert.assertTrue( MatrixOps.allclose( x, y ) );
  }

  @Test
	public void rowNormalize() {
    double[][] X1_ = {
      { 1.74702881, -1.11241135,  0.18658948},
      { 0.0449732 , -0.16616407,  1.12119088},
      { 0.95235621, -0.81554199, -0.33258448}};

    SimpleMatrix X1a = new SimpleMatrix( X1_ );
    SimpleMatrix X1b = new SimpleMatrix(X1);
    MatrixOps.rowNormalize( X1b.getMatrix(), 1 );
    Assert.assertTrue( MatrixOps.allclose( X1a, X1b ) );
  }
	
  @Test
	public void columnNormalize() {
    double[][] X1_ = {
      { 1.74702881,  0.65171457,  0.18658948},
      {-0.05982937, -0.12950598, -1.49155831},
      { 0.95235621,  0.47779142, -0.33258448}};

    SimpleMatrix X1a = new SimpleMatrix( X1_ );
    SimpleMatrix X1b = new SimpleMatrix(X1);
    MatrixOps.columnNormalize( X1b.getMatrix(), 1 );

    Assert.assertTrue( MatrixOps.allclose( X1a, X1b ) );
  }

  @Test
	public void cdist() {
    double[][] D_ = {
      { 4.6537912 ,  1.48663463,  2.78263492,  2.37496883,  2.54702606,
         1.74098855},
       { 1.8519458 ,  2.3032351 ,  3.24306133,  1.9171944 ,  0.69296655,
         1.06318246},
       { 3.69769277,  0.98856228,  2.41459114,  1.85265058,  1.70587533,
         0.81501276}};

    SimpleMatrix Da = new SimpleMatrix( D_ );
    SimpleMatrix Db = MatrixOps.cdist( X1, X4 );
    Assert.assertTrue( MatrixOps.allclose( Da, Db ) );
  }

  @Test
	public void projectOntoSimplexArray() {
    double[] theta1 = { 0.50481492,  0.47901866,  0.01616642 };
    double[] theta2 = { 0.50481492,  0.47901866,  -0.01616642 };
    double[] theta1A = { 0.50481492,  0.47901866,  0.01616642 };
    double[] theta2A = { 0.51311007,  0.48688993,  0.         };

    MatrixOps.projectOntoSimplex(theta1);
    MatrixOps.projectOntoSimplex(theta2);

    Assert.assertTrue( MatrixOps.allclose( theta1, theta1A) );
    Assert.assertTrue( MatrixOps.allclose( theta2, theta2A) );
  }

  @Test
	public void projectOntoSimplex() {
    double[][] theta_ = { 
      { 0.50481492,  0.47901866,  0.01616642 },
      { 0.50481492,  0.47901866,  -0.01616642 }};
    double[][] thetaA_ = {
      { 0.50481492,  0.47901866,  0.01616642 },
      { 0.51311007,  0.48688993,  0.         }};

    SimpleMatrix theta = new SimpleMatrix( theta_ );
    SimpleMatrix thetaA = new SimpleMatrix( thetaA_ );

    SimpleMatrix thetaB = MatrixOps.projectOntoSimplex( thetaA.transpose() ).transpose();

    Assert.assertTrue( MatrixOps.allclose( thetaA, thetaB ) );
  }

  @Test
	public void rank() {
    double[][] R1Matrix_ = {
      { 0.39862939,  0.37825926,  0.01276589},
      { 0.37825926,  0.35893005,  0.01211354},
      { 0.01276589,  0.01211354,  0.00040882}};
    SimpleMatrix R1Matrix = new SimpleMatrix( R1Matrix_ );

    Assert.assertTrue( MatrixOps.rank( X1 ) == 3 );
    Assert.assertTrue( MatrixOps.rank( R1Matrix ) == 1 );
  }

  @Test
	public void svdk() {
    double[][] U_ = {
      {-0.85615465, -0.02776278},
      { 0.11303394,  0.96430814},
      {-0.50420486,  0.26332308}};
    double[][] W_ = {
      { 2.42589405,  0.0 },
      {  0.0, 1.53816955 }};
    double[][] V_ = {
      {-0.81729547,  0.09399552}, 
      { 0.5724005 ,  0.01904634},
      { -0.06622525, -0.99539042,}};

    SimpleMatrix U = new SimpleMatrix( U_ );
    SimpleMatrix W = new SimpleMatrix( W_ );
    SimpleMatrix V = new SimpleMatrix( V_ );

    Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix> UWV = MatrixOps.svdk( X1, 2 );
    Assert.assertTrue( MatrixOps.allclose( MatrixOps.abs( U ), MatrixOps.abs( UWV.getValue0() ) ) );
    Assert.assertTrue( MatrixOps.allclose( MatrixOps.abs( W ), MatrixOps.abs( UWV.getValue1() ) ) );
    Assert.assertTrue( MatrixOps.allclose( MatrixOps.abs( V ), MatrixOps.abs( UWV.getValue2() ) ) );
  }

  @Test
	public void approxk() {
    double[][] X1_R2Approx_ = {
      { 1.69346008, -1.18965511,  0.18005292},
      {-0.08468856,  0.18520784, -1.49459168},
      { 1.03774449, -0.69241583, -0.32216523}};
    SimpleMatrix X1_R2ApproxA = new SimpleMatrix( X1_R2Approx_ );

    SimpleMatrix X1_R2ApproxB = MatrixOps.approxk( X1, 2 );
    Assert.assertTrue( MatrixOps.allclose( X1_R2ApproxA, X1_R2ApproxB ) );
  }

  @Test
	public void eig() {
    double[][] L_ = {{-0.94770034, 0.31734053, 2.26585785}};
    double[][] R_ = { 
      { 0.2740856 ,  0.61844862,  0.84960229},
      { 0.76229792,  0.78220896, -0.32681905},
      { 0.58632667, -0.07530232,  0.41396288}};

    SimpleMatrix L = new SimpleMatrix(L_);
    SimpleMatrix R = new SimpleMatrix(R_);

    try {
      SimpleMatrix[] LR = MatrixOps.eig( X1 );
      Assert.assertTrue( MatrixOps.allclose( L, LR[0] ) );
      Assert.assertTrue( MatrixOps.allclose( MatrixOps.abs( R ), MatrixOps.abs( LR[1] ) ) );
    } catch( NumericalException e ) {
      Assert.fail( e.getMessage() );
    }
  }

  @Test
	public void equals() {
    Assert.assertTrue( MatrixOps.equal( 1.0, 0.9999999 ) );
    Assert.assertFalse( MatrixOps.equal( 1.0, 0.99 ) );
  }

  @Test
  public void rowStack() {
    double[][] X_ = {
            { 0.2740856 ,  0.61844862,  0.84960229},
            { 0.76229792,  0.78220896, -0.32681905},
            { 0.58632667, -0.07530232,  0.41396288}};
    SimpleMatrix X = new SimpleMatrix(X_);

    SimpleMatrix[] Xrows = new SimpleMatrix[3];
    for( int i = 0; i < 3; i++ )
      Xrows[i] = MatrixFactory.fromVector(X_[i]);
    SimpleMatrix Y = MatrixFactory.rowStack(Xrows);
    SimpleMatrix Z = MatrixFactory.columnStack(Xrows);

    Assert.assertTrue( MatrixOps.allclose( X, Y ) );
    Assert.assertTrue( MatrixOps.allclose( X.transpose(), Z ) );
  }

  @Test
  public void whitener() {
    for( int i = 0; i < 3; i++ ) {
      SimpleMatrix X = RandomFactory.randn( 5, 3 );
      SimpleMatrix Y = X.mult(X.transpose());

      SimpleMatrix W = MatrixOps.whitener(Y);
      SimpleMatrix Z = W.transpose().mult(Y).mult(W);

      Assert.assertTrue( MatrixOps.allclose( Z, SimpleMatrix.identity(3) ) );
    }
  }

  @Test
  public void colorer() {
    for( int i = 0; i < 3; i++ ) {
      SimpleMatrix X = RandomFactory.randn( 3, 3 );
      SimpleMatrix Y = X.mult(X.transpose());

      SimpleMatrix W = MatrixOps.whitener(Y);
      SimpleMatrix Winv = MatrixOps.colorer(Y);

      Assert.assertTrue( MatrixOps.allclose( W.transpose().mult(Winv), SimpleMatrix.identity(3) ) );
    }
  }


  @Test
  public void isSymmetric() {
    double[][] X_ = {
            { 0.2740856 ,  0.61844862,  0.84960229},
            { 0.76229792,  0.78220896, -0.32681905},
            { 0.58632667, -0.07530232,  0.41396288}};
    SimpleMatrix X = new SimpleMatrix(X_);
    SimpleMatrix Y = X.plus( X.transpose() );

    Assert.assertFalse( MatrixOps.isSymmetric( X ) );
    Assert.assertTrue( MatrixOps.isSymmetric( Y ) );

    Assert.assertFalse( MatrixOps.isSymmetric( T ) );
    Assert.assertTrue( MatrixOps.isSymmetric( symT ) );
  }

}
