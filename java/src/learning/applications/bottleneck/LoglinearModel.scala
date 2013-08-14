package learning.applications.bottleneck

/**
 * A log-linear model
 *    - has features phi(x,y)
 *    - has parameters theta
 */
abstract class LogLinearModel extends Optimizable {
  val features : Map[String,Int]
//  val parameters : Vector[Double]

  def likelihood : Vector[Option[Double]] => Double
  def size = features.size
}

trait Optimizable {
  def value : Vector[Double] => Double
  def gradient : Vector[Double] => Vector[Double]
}

