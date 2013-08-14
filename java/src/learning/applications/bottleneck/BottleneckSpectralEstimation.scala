package learning.applications.bottleneck

import breeze.linalg.DenseVector
import breeze.stats.distributions.Rand
import scala.util.Random

/**
 * Parameter estimation for general graphical models by estimating conditional moments of "bottlenecks".
 *     - Uses optimization to solve for parameters.
 *     - Uses the tensor power method to recover these conditional moments.
 */
class BottleneckSpectralEstimation {
  val rnd = new Random(1)

  def optimize (model : LogLinearModel, data : List[Vector[Double]]) = {
    val beta = DenseVector.rand(model.size, rnd) // Note that in our realization, beta and theta share features
    val theta = DenseVector.rand(model.size, rnd)

    // Optimize L(\theta, \beta) with respect to \beta first.


    ()
  }

}
