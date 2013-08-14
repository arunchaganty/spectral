package learning.applications.bottleneck
import breeze.linalg._

/**
 * Implements a simple (undirected) mixture model as a log-linear model
 *  - has features I[h=?] and I[h=?,x=?]
 */
class MixtureModel(k:Int, d: Int) extends LogLinearModel {
  val features : Map[String,Int] = {
    def idx_h(i : Int) = s"y=$i" -> i
    def idx_hx(tpl : (Int,Int)) = tpl match { case (i:Int,j:Int) => s"y=$i,x=$j" -> (k + d*i + j) }

    val hidden = List.range[Int](0,k)
    val observed = List.range[Int](0,d)
    (hidden.map(idx_h) ++ hidden.cross(observed).map(idx_hx)).toMap
  }

  implicit class Crossable[X](xs: Traversable[X]) {
    def cross[Y](ys: Traversable[Y]) = for { x <- xs; y <- ys } yield (x, y)
  }

  val parameters: Vector[Double] = DenseVector.zeros[Double](features.size)

  def likelihood: (Vector[Option[Double]]) => Double = params => 0.0
  def value: Vector[Double] => Double = params => this.likelihood(params.map(v => Option(v)))
  def gradient: Vector[Double] => Vector[Double] = params => DenseVector.zeros(features.size)
}

object testObject {
  def main() = {
    val model = new MixtureModel(2,2)
    model.features
  }
}
