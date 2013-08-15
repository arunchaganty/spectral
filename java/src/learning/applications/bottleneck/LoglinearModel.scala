package learning.applications.bottleneck
import breeze.linalg._

/**
 * A log-linear model
 *    - has features phi(x,y)
 *    - has parameters theta
 */
abstract class LogLinearModel { //extends Optimizable {
  /**
   * Stores a map of all the features in this model.
   */
  val features : Map[String,Int]

  /**
   * @return Size of the model = number of features = number of parameters
   */
  def size = features.size

  /**
   * @return The empty parameter set
   */
  def zeroParameter = DenseVector.zeros[Double](size)

  /**
   * Computes the likelihood of the data given this set of parameters.
   * @param params - The parameters of the model
   * @param data - Data who's likelihood is to be computed
   * @return - Likelihood of data
   */
  def likelihood( params : Vector[Double], data : HashVector[Double] ) : Double

//  def likelihood( params : Vector[Double], data : HashMatrix[Double] ) =
//    List.range(0,data.cols).reduceLeft( (lhood:Double, col) => lhood + likelihood(params, data(::,col) : Vector[Option[Double]] ) )

  /**
   * Computes the marginal probability of the unobserved variables
   * @param params - parameters of the model
   * @param data - partially observed data
   * @return - fully observed data
   */
  def marginal( params : Vector[Double], data : Vector[Option[Double]] ) : Vector[Double]

//  /**
//   * Computes the argmax assignment of the unobserved variables
//   * @param params - parameters of the model
//   * @param data - partially observed data
//   * @return - fully observed data (using argmax)
//   */
  //def viterbi( params : Vector[Double], data : Vector[Option[Double]] ) : Vector[Double]
}

trait Optimizable {
  def value( params : Vector[Double] ) : Double
  def gradient( params : Vector[Double] ) : Vector[Double]
}

