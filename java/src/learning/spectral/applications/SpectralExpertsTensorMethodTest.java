/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral.applications;

import learning.exceptions.RecoveryFailure;
import learning.exceptions.NumericalException;

import learning.linalg.*;
import learning.models.MixtureOfExperts;

import learning.optimization.PhaseRecovery;
import learning.optimization.ProximalGradientSolver;
import learning.optimization.TensorRecovery;
import learning.spectral.MultiViewMixture;

import learning.data.MomentComputer;
import learning.data.RealSequence;

import learning.spectral.TensorMethod;
import org.ejml.alg.dense.mult.GeneratorMatrixMatrixMult;
import org.ejml.alg.dense.mult.MatrixMatrixMult;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.simple.SimpleBase;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.*;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OptionsParser;
import fig.exec.Execution;

import java.lang.ref.SoftReference;
import java.util.Date;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.ClassNotFoundException;

/**
 * Spectral Experts.
 */
public class SpectralExpertsTensorMethodTest implements Runnable {
	@Option(gloss = "Number of classes")
	public int K = 0;
	@Option(gloss = "Input filename", required=true)
	public String inputPath;

  @Option(gloss = "Error to be added")
  public double epsilon = 1e-1;

  SpectralExperts.SpectralExpertsAnalysis analysis;

  /**
   * Run the SpectralExperts algorithm on data perturbed by some epsilon
   * @param y
   * @param X
   * @return
   * @throws NumericalException
   */
  public Pair<SimpleMatrix, SimpleMatrix> run(MixtureOfExperts model) throws NumericalException, RecoveryFailure {
    int D = model.getDataDimension();
    SimpleMatrix Pairs = analysis.Pairs;
    FullTensor Triples = analysis.Triples;
    SimpleMatrix err = RandomFactory.rand( D, D );
    FullTensor errT = RandomFactory.uniformTensor( D, D, D );

    SimpleMatrix Pairs_ = analysis.Pairs.plus( epsilon/(D*D), err );
    FullTensor Triples_ = analysis.Triples.plus( epsilon/(D*D*D), errT );

    analysis.reportPairs( Pairs_ );
    analysis.reportTriples( Triples_ );

    // Use the tensor power method to recover $\betas$.
    TensorMethod algo = new TensorMethod();
    Pair<SimpleMatrix, SimpleMatrix> pair = algo.recoverParameters( K, Pairs_, Triples_ );
    // Somewhat of a "hack" to try and rescale the weights to sum to 1
    SimpleMatrix weights = pair.getValue0();
    SimpleMatrix betas = pair.getValue1();

    // Normalize the weights at the very least
    double sum = weights.elementSum();
    weights = weights.scale( 1/sum );

    analysis.reportWeights(weights);
    analysis.reportBetas(betas);

    return new Pair<>(weights, betas);
  }

	@Override
	public void run() {
    try {
      // Read data from a file
      Pair< Pair<SimpleMatrix, SimpleMatrix>, learning.models.MixtureOfExperts > data =
              MixtureOfExperts.readFromFile( inputPath );
      learning.models.MixtureOfExperts model = data.getValue1();
      analysis = new SpectralExperts.SpectralExpertsAnalysis( model );

      data = null;

      // Set K from the model if it hasn't been provided
      if( K < 1 )
        K = model.getK();
      int D = model.getDataDimension();

      Pair<SimpleMatrix, SimpleMatrix> pi_betas_ = run( model );

      SimpleMatrix weights_ = pi_betas_.getValue0();
      SimpleMatrix betas_ = pi_betas_.getValue1();
      System.out.printf( "%.4f\n", analysis.betasErr );
    } catch( ClassNotFoundException | IOException |  NumericalException | RecoveryFailure e ) {
      System.err.println( e.getMessage() );
      return;
    }
	}

	/**
	 * Mixture of Linear Regressions
	 * @param args
	 * @throws IOException
	 * @throws RecoveryFailure 
	 */
	public static void main(String[] args) throws IOException, RecoveryFailure {
		Execution.run( args, new SpectralExpertsTensorMethodTest() );
	}
}
