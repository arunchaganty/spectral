/**
 * learning.spectral
 * Arun Chaganty <chaganty@stanford.edu
 *
 */
package learning.em;

import learning.Misc;
import learning.linalg.*;
import learning.data.*;
import learning.spectral.TensorMethod;
import learning.models.loglinear.*;
import learning.models.loglinear.Models.HiddenMarkovModel;

import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

import fig.basic.*;
import fig.exec.*;
import fig.prob.*;
import fig.record.*;
import static fig.basic.LogInfo.*;

import java.io.*;
import java.util.*;

/**
 * Learns EM via optimization.
 * For likelihoods like \theta^T E_q
 */
public interface EMOptimizable {

  public int numFeatures();
  // L = \theta^T \E_q, gradient = \E_q
  public void setParams(double[] params);
  public double compute(int[][] X, double[] gradient);

}

