package learning.applications.bottleneck

import breeze.linalg.DenseVector
import breeze.linalg._
import scala.util.Random
import breeze.optimize.DiffFunction
import breeze.math._
import org.scalatest.{BeforeAndAfter, FunSpec}

/**
 * Parameter estimation for general graphical models by estimating conditional moments of "bottlenecks".
 *     - Uses optimization to solve for parameters.
 *     - Uses the tensor power method to recover these conditional moments.
 */
class BottleneckSpectralEstimation {
  val rnd = new Random(1)

  /**
   * MeasurementsEM objective, as taken from
   *    - "Learning from measurements in exponential families"; Percy Liang, Michael I. Jordan, Dan Klein.
   * $L(\beta, \theta | \tau, \phi, \sigma) =
   *  <\tau, \beta> - \sum_{i=1}^^n B(\beta, \theta, X_i)
   *  + \sum_{i=1}^^n A(\theta, X_i) - h^^*_{\sigma}(\beta) + h_{\phi}(\theta).$
   */
  class MeasurementEObjective(theta:DenseVector[Double], model : MixtureModel, tau : DenseVector[Double], data : Vector[Vector[Int]]) extends DiffFunction[DenseVector[Double]] {

    def calculate(beta: DenseVector[Double]): (Double, DenseVector[Double]) = {
      val (a: Double) = sum( data.map( (datum : Vector[Int]) => model.marginalLogLikelihood(theta, datum):Double ) )
      val (b : Double) = sum( data.map( (datum : Vector[Int]) => model.marginalLogLikelihood(beta, datum)) )
      val l = tau.dot(beta) - b + a - 0.5 * Math.pow(beta.norm(2),2) + 0.5 * Math.pow(theta.norm(2),2)
      val counts = data.valuesIterator.foldLeft(DenseVector.zeros[Double](beta.length))( (acc : DenseVector[Double], datum: Vector[Int]) => acc + model.posterior(theta, datum) )
      val (grad : DenseVector[Double]) = tau - counts - beta

      (l, grad)
    }
  }

  def optimize (model : LogLinearModel, data : List[Vector[Double]]) = {
    val beta = DenseVector.rand(model.size, rnd) // Note that in our realization, beta and theta share features
    val theta = DenseVector.rand(model.size, rnd)
  }
}

/**
 * EM objective:
 * $L(\theta) = \sum_{i=1}^^n A(\theta, X_i)$
 */
class EMObjective[U <: Vector[ DenseVector[Int]]](model : MixtureModel, data : U) extends DiffFunction[DenseVector[Double]] {
  def calculate(theta: DenseVector[Double]): (Double, DenseVector[Double]) = {
    val (l:Double) =  data.map( (datum : Vector[Int]) => model.marginalLogLikelihood(theta, datum) ).toDenseVector.sum
//    val counts = data.valuesIterator.foldLeft(DenseVector.zeros[Double](model.hidden.length))( (acc : DenseVector[Double], datum: Vector[Int]) => acc + model.posterior(theta, datum) )
//    val (grad : DenseVector[Double]) = counts
    val (grad : DenseVector[Double]) = DenseVector.zeros[Double](model.features.size)
    (l, grad)
  }
}

class EMObjectiveTest extends FunSpec {
  val N = 100
  val rnd = new Random(1)

  val model : MixtureModel = new MixtureModel(2,2,1)
  val params = DenseVector.rand( model.features.size, rnd )
  val data : DenseVector[DenseVector[Int]] = model.generate(params, N)
  val observedData = MixtureModel.projectObserved(data)

  val obj = new EMObjective(model, observedData)

  describe("An EM objective") {
    it( "should pass the gradient check" ) {
      def unit(size:Int, i:Int, value:Double = 1.) = {
          val v = DenseVector.zeros[Double](size)
          v(i) = value
          v
        }

      val actualGradient = obj.gradientAt(params)

      val eps = 1e-5
      for {d <- List.range(0,params.size)} {
        val expectedGradient = {
          (obj.valueAt(params + unit(params.size, d, eps)) -  obj.valueAt(params + unit(params.size, d, -eps)))/2*eps
        }
        assert( Math.abs(actualGradient(d) - expectedGradient) < eps )
      }
    }
    it( "should find the exact parameters for a simple model") {
    }
    it( "should have a monotonic objective" ) {
    }
  }
}

class MeasurementsObjectiveTest extends FunSpec {
  describe("An measurement objective") {
    it( "should pass the gradient check" ) {
    }
    it( "should find the exact parameters for a simple model") {
    }
    it( "should have a monotonic objective" ) {
    }
  }
}
