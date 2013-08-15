package learning.applications.bottleneck
import learning.utils.Utils._
import breeze.linalg._
import org.scalatest._

/**
 * Implements a simple (undirected) mixture model as a log-linear model
 *  - has features I[h=?] and I[h=?,x=?]
 */
class MixtureModel(k:Int, d: Int, v:Int) {
  val variables : Map[String,Int] =
    (("y" -> 0) :: List.range(0,v).map( (i:Int) => s"x_$i" -> (i+1) )).toMap
  val features : Map[String,Int] = {
    def idx_h(i : Int) = s"y=$i" -> i
    def idx_hx(tpl : (Int,Int,Int)) = tpl match { case (i:Int,j:Int,v:Int) => s"y=$i,x_$v=$j" -> (k + (d*k)*v + d*i + j) }

    val hidden = List.range[Int](0,k)
    val observed = List.range[Int](0,d)
    val views = List.range[Int](0,v)
    (hidden.map(idx_h) ++ hidden.cross3(observed, views).map(idx_hx)).toMap
  }

  def Z(params: Vector[Double]) = {
    1.0
  }

  /**
   * Compute the marginal likelihood of each observed state
   * @param params
   * @param observed
   */
  def marginalLikelihood( params : Vector[Double], observed : Vector[Int] ) : Vector[Double] = {
    val (full : DenseVector[Int]) = DenseVector.zeros[Int](variables.size)
    full.slice(1,full.length) := observed
    val (lhoods : List[Double]) =  List.range(0,k).map(i => { full(0) = i; likelihood(params, full) } )
    val norm = logsumexp(lhoods)
    DenseVector[Double]( lhoods.map( lhood => lhood - norm ).toArray : Array[Double] )
  }

  def likelihood( params : Vector[Double], full : Vector[Int] ) = {
    // Compute the likelihood by adding the right parameters
    val yi = full(variables("y"))
    def yx_idx(i:Int) : String = s"y=$yi,x_$i="+ full(variables(s"x_$i"))
    List.range(0,v).foldLeft(params( features(s"y=$yi")))( (acc:Double,i:Int) => acc + params( features(yx_idx(i)) ) )
  }
  def marginal( params : Vector[Double], observed : Vector[Double] ) = DenseVector.zeros[Double](1)
}

class MixtureModelTest extends FunSpec {
  describe("A mixture model") {
    it( "should have 1 + v variables" ) {
      for { tpl <- List((2,2,1), (2,2,3), (2,3,4)) }
        tpl match { case (k,d,v) =>  assert( new MixtureModel(k,d,v).variables.size == 1 + v ) }
    }
    it( "should have k + k*d*v features" ) {
      for { tpl <- List((2,2,1), (2,2,3), (2,3,4)) }
        tpl match { case (k,d,v) =>  assert( new MixtureModel(k,d,v).features.size == k + v*(k*d) ) }
    }
    it( "should evaluate the likelihood" ) {
      val model = new MixtureModel(2,2,3)
      assert( model.likelihood( DenseVector.zeros[Double](model.features.size), DenseVector.zeros[Int](model.variables.size) ) == 0.0 )
    }
    it( "should evaluate the likelihood" ) {
      val model = new MixtureModel(2,2,3)
      val pr = model.marginalLikelihood( DenseVector.zeros[Double](model.features.size), DenseVector.zeros[Int](model.variables.size-1) )
      pr.map( p => Math.exp(p) )
    }
  }
}



