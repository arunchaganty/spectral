package learning.linalg;

import fig.basic.*;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;

import java.util.Random;

/**
 * Implements Anandkumar/Ge/Hsu/Kakade/Telgarsky, 2012.
 * Method of moments for learning latent-variable models.
 */
public class TensorPowerMethod extends DeflatingTensorFactorizationAlgorithm {
  static final double EPS_CLOSE = 1e-10;
  @Option(gloss="Number of iterations to run the power method")
  public int iters = 100;
  @Option(gloss="Number of attempts in order to find the real maxima")
  public int attempts = 100;
  @Option(gloss="Random number generator for tensor method and random projections")
  Random rnd = new Random();

  public TensorPowerMethod() {
  }
  public TensorPowerMethod(int iters, int attempts) {
    this.iters = iters;
    this.attempts = attempts;
  }

  /**
   * Perform a single eigen-decomposition step
   * @param T - Full tensor
   * @return - (eigenvalue, eigenvector).
   */
  protected Pair<Double,SimpleMatrix> eigendecomposeStep( FullTensor T ) {
    LogInfo.begin_track("PowerMethod:step");

    int N = iters;
    int D = T.getDim(0);

    double maxEigenvalue = Double.NEGATIVE_INFINITY;
    SimpleMatrix maxEigenvector = null;

    for( int attempt = 0; attempt < attempts; attempt++ ) {
      // 1. Draw theta randomly from unit sphere
      SimpleMatrix theta = RandomFactory.rand(rnd, 1, D);
      theta.scale(1.0/MatrixOps.norm(theta));
      if( attempt % 10 == 0 )
        LogInfo.dbgs("Attempt %d/%d", attempt, attempts);

      // 2. Compute power iteration update
      for(int n = 0; n < N; n++ ) {
        SimpleMatrix theta_ = T.project2(1, 2, theta, theta);
        // Normalize
        theta_ = theta_.scale(1.0 / MatrixOps.norm(theta_));
        double err = MatrixOps.norm(theta_.minus(theta));
        theta = theta_;
        if( err < EPS_CLOSE ) break;
      }
      double eigenvalue = T.project3(theta, theta, theta);
      if(eigenvalue > maxEigenvalue) {
        maxEigenvalue = eigenvalue;
        maxEigenvector = theta;
      }
    }
    SimpleMatrix theta = maxEigenvector;
    // 3. Do N iterations with this max eigenvector
    for(int n = 0; n < N; n++ ) {
      SimpleMatrix theta_ = T.project2(1, 2, theta, theta);
      // Normalize
      theta_ = theta_.scale(1.0 / MatrixOps.norm(theta_));
      double err = MatrixOps.norm(theta_.minus(theta));
      theta = theta_;
      if( err < EPS_CLOSE ) break;
    }
    double eigenvalue = T.project3(theta, theta, theta);
    LogInfo.end_track("PowerMethod:step");

    return Pair.with(eigenvalue, theta);
  }

}
