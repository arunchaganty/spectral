package learning.optimization;

import fig.basic.LogInfo;
import learning.linalg.FullTensor;
import learning.linalg.MatrixOps;
import learning.linalg.RandomFactory;
import learning.models.MixtureOfExperts;
import org.ejml.data.DenseMatrix64F;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;
import org.junit.Assert;
import learning.optimization.ProximalGradientSolver.LearningRate;
import org.junit.Test;

public class ProximalGradientDescentTest {

  @Test
  public void regression() {
    // Use a single mixture to generate a regression problem to solve.
    MixtureOfExperts.GenerationOptions options = new MixtureOfExperts.GenerationOptions();
    options.D = 3; options.K = 1; options.betas = "random"; options.weights = "random";
    MixtureOfExperts model = MixtureOfExperts.generate(options);
    Pair<SimpleMatrix, SimpleMatrix> yX = model.sample(1000);
    final DenseMatrix64F y = yX.getValue0().getMatrix();
    final DenseMatrix64F X = yX.getValue1().getMatrix();
    final int N = X.numRows;
    final int D = X.numCols;

    // Define a simple problem.
    DenseMatrix64F initialState = new DenseMatrix64F(1, D);
    DenseMatrix64F betas = model.getBetas().transpose().getMatrix();

    ProximalGradientSolver solver = new ProximalGradientSolver();
    DenseMatrix64F betas_ = solver.optimize(new Regression(y, X), initialState, new LearningRate(LearningRate.Type.BY_SQRT_T, 0.99), 100, 1e-6);

    System.out.println( betas_ );
    System.out.println( betas );
    LogInfo.logs( "Error: " + MatrixOps.norm( SimpleMatrix.wrap( betas ).minus( SimpleMatrix.wrap(betas_))));

    Assert.assertTrue(MatrixOps.allclose( betas, betas_, 1e-2));
  }

  @Test
  public void phaseRecovery() {
    final int N = (int) 1e3; final int D = 20; final int K = 2; final double reg = 1.8e-3;
    // General phase recovery.
    // Create a rank 2 matrix
    SimpleMatrix M = RandomFactory.randn( D, D );
    // Symmetrize
    M = M.plus(M.transpose()).scale(0.5);
    M = MatrixOps.approxk( M, K );

    // Create some data points
    SimpleMatrix X = RandomFactory.rand(N, D);
    // Get some points
    SimpleMatrix y = MatrixOps.quadraticForm( X, M );
    // Add a little noise
    y = y.plus( 0.05, RandomFactory.randn(N, 1) );

    ProximalGradientSolver solver = new ProximalGradientSolver();

    // Optimize with just regression
    SimpleMatrix initialState = RandomFactory.randn(D,D).scale(0.01);
    initialState = initialState.plus(initialState.transpose()).scale(0.5);

    ProximalGradientSolver.ProximalOptimizable problem = new PhaseRecovery(y, X, reg);
    DenseMatrix64F M_ = solver.optimize(problem, initialState.getMatrix(),
            new LearningRate(LearningRate.Type.BY_SQRT_T, 2.99),
            400, 1e-6);
    LogInfo.logs( "Solution had rank: " + MatrixOps.rank(M_) + " instead of " + M.svd().rank() );
    LogInfo.logs( "Error: " + MatrixOps.diff( M.getMatrix(), M_ ) );
    Assert.assertTrue(MatrixOps.allclose( M.getMatrix(), M_, 1e-1));

    M_ = MatrixOps.approxk( SimpleMatrix.wrap( M_ ), K ).getMatrix();
    LogInfo.logs( "Error after reducing rank to K: " + MatrixOps.diff( M.getMatrix(), M_ ) );
  }

  @Test
  public void tensorRecovery() {
    final int N = (int) 1e3; final int D = 3; final int K = 2; final double reg = 6e-4;
    // General phase recovery.
    // Create a rank K tensor
    FullTensor T = RandomFactory.symmetricTensor(K, D);
    DenseMatrix64F M = T.unfold(0).getMatrix();

    // Create some data points
    SimpleMatrix X = RandomFactory.rand(N, D);
    DenseMatrix64F x = new DenseMatrix64F(1,D);
    // Get some points
    SimpleMatrix y = new SimpleMatrix(N, 1);
    for(int n = 0; n < N; n++) {
      MatrixOps.row(X.getMatrix(), n, x);
      y.set(n, T.project3(x, x, x));
    }
    // Add a little noise
    y = y.plus( 0.05, RandomFactory.randn(N, 1) );

    ProximalGradientSolver solver = new ProximalGradientSolver();

    // Optimize with just regression
    SimpleMatrix initialState = RandomFactory.randn(1,D).scale(0.01);
    initialState = FullTensor.fromUnitVector(initialState).unfold(0);

    ProximalGradientSolver.ProximalOptimizable problem = new TensorRecovery(y, X, reg);
    DenseMatrix64F M_ = solver.optimize(problem, initialState.getMatrix(),
            new LearningRate(LearningRate.Type.BY_SQRT_T, 2.99),
            400, 1e-6);
    LogInfo.logs( "Solution had rank: " + MatrixOps.rank(M_) + " instead of " + MatrixOps.rank(M) );
    LogInfo.logs( "Error: " + MatrixOps.diff( M, M_ ) );
    Assert.assertTrue(MatrixOps.allclose( M, M_, 1e-1));

    // TODO: This isn't really kosher
    M_ = MatrixOps.approxk( SimpleMatrix.wrap( M_ ), K ).getMatrix();
    LogInfo.logs( "Error after reducing rank to K: " + MatrixOps.diff( M, M_ ) );
  }
}
