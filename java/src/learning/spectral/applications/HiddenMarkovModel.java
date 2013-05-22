/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral.applications;

import learning.spectral.TensorMethod;
import learning.data.Corpus;
import learning.data.ProjectedCorpus;
import learning.data.MomentComputer;
import learning.linalg.*;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.*;

import fig.basic.LogInfo;
import fig.basic.Option;
import fig.basic.OptionsParser;
import fig.exec.Execution;

import java.util.Date;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.ClassNotFoundException;

/**
 * Word clustering using a HMM model
 *    + Reads data as a sequence of real-valued vectors, or a moment matrix
 *      of the same
 *    + Runs tensor-factorization and does post processing to find
 *    emission and transition probabilities.
 */
public class HiddenMarkovModel implements Runnable {

  public Triplet<SimpleMatrix, SimpleMatrix, SimpleMatrix>
      run( int K, 
          SimpleMatrix M12, SimpleMatrix M13, SimpleMatrix M23, FullTensor M123 ) {
    // Tensor Factorize
    TensorMethod algo = new TensorMethod();
    Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, SimpleMatrix> params = algo.recoverParameters( K, M12, M13, M23, M123 );

    // Return emission and transition probabilities.

    return null;
  }

  public void run() {}

}

