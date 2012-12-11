/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.linalg;

import learning.linalg.MatrixOps;

import org.ejml.simple.SimpleMatrix;
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

    X1 = new SimpleMatrix(X1_);
    X2 = new SimpleMatrix(X2_);
    X3 = new SimpleMatrix(X3_);
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

    SimpleTensor P123_ = MatrixOps.Triples( X1, X2, X3 );
    SimpleMatrix P123_1a = P123_.project( 2, theta1 );
    SimpleMatrix P123_2a = P123_.project( 2, theta2 );

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
  }

  @Test
	public void setRow() {
    double[][] theta1 = {{ 0.63137104,  0.59910771,  0.02021931}};
    double[][] theta2 = {{ 0.82945477,  0.23272397,  0.60360884}};

    double[][] X1a_ = {
      { 1.74702881, -1.11241135,  0.18658948},
      { 0.63137104,  0.59910771,  0.02021931},
      { 0.95235621, -0.81554199, -0.33258448}};

    SimpleMatrix X1a = new SimpleMatrix( X1a_ );
    SimpleMatrix X1b = new SimpleMatrix(X1);
    MatrixOps.setRow( X1b, 1, new SimpleMatrix(theta1) );

    Assert.assertTrue( MatrixOps.allclose( X1a, X1b ) );
  }
	
  @Test
	public void setCols() {
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
	public void rowSum() {
  }
	
  @Test
	public void columnSum() {
  }


  @Test
	public void rowNormalize() {
  }
	
  @Test
	public void columnNormalize() {
  }

  @Test
	public void cdist() {
  }

  @Test
	public void projectOntoSimplex() {
  }

  @Test
	public void rank() {
  }

  @Test
	public void svdk() {
  }

  @Test
	public void approxk() {
  }

  @Test
	public void eig() {
  }

}
