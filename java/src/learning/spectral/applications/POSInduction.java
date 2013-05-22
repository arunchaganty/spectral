/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.spectral.applications;

import learning.exceptions.RecoveryFailure;
import learning.exceptions.NumericalException;

import learning.Misc;
import learning.linalg.*;
import learning.models.MixtureOfExperts;

import learning.optimization.PhaseRecovery;
import learning.optimization.ProximalGradientSolver;
import learning.optimization.ProximalGradientSolver.LearningRate;
import learning.optimization.TensorRecovery;
import learning.optimization.MatlabProxy;
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
import java.lang.ClassNotFoundException;
import java.io.*;

/**
 * Perform POS induction, aka HMM learning.
 */
public class POSInduction implements Runnable {

  // Read data as word-index sequences
  // Project to random features using random-map
  // Run HMM model
  // Return unprojected emission and transition probabilities.
  
  public void run() {}

}


