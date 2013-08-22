package learning.applications.bottleneck
import learning.utils.Utils._
import breeze.linalg._
import org.scalatest._
import breeze.stats.distributions.{DiscreteDistr, Multinomial}
import scala.util.Random

/**
 * Implements a simple (undirected) mixture model as a log-linear model
 *  - has features I[h=?] and I[h=?,x=?]
 */
class MixtureModel(K:Int, D: Int, V:Int) {
  var rnd = new Random(1)

  val variables : Map[String,Int] =
    (("y" -> 0) :: List.range(0,V).map( (i:Int) => s"x_$i" -> (i+1) )).toMap
  val features : Map[String,Int] = {
    def idx_h(i : Int) = s"y=$i" -> i
    def idx_hx(tpl : (Int,Int,Int)) = tpl match { case (i:Int,j:Int,v:Int) => s"y=$i,x_$v=$j" -> (K + (D*K)*v + D*i + j) }

    val hidden = List.range[Int](0,K)
    val observed = List.range[Int](0,D)
    val views = List.range[Int](0,V)
    (hidden.map(idx_h) ++ hidden.cross3(observed, views).map(idx_hx)).toMap
  }

  def toDistribution(params: Traversable[Double]) = {
    val (norm : Double) = logsumexp(params)
    val probs = DenseVector( params.map( (param:Double) => Math.exp( param - norm ) ).toArray )
    new Multinomial[Vector[Double],Int](probs)
  }

  def generate[T <: Vector[Double]](params : T) : Vector[Int] = {
    generate(params,1)(0)
  }
  def generate[T <: Vector[Double]](params : T, N : Int) : DenseVector[DenseVector[Int]] = {
    def params_(k:Int, d:Int, v:Int) = params(features(s"y=$k,x_$v=$d"))
    // Compute distributions from the parameters by marginalizing out various indices
    val yDist = toDistribution(
        List.range(0,K).map( k => {
          params(k) +
            List.range(0,D).cross(List.range(0,V)).foldLeft(0.)(
            (acc:Double, z:(Int,Int)) => acc + params_(k, z._1,z._2) )
          }
        )
    )
    val (xDist : Map[(Int,Int),DiscreteDistr[Int]]) = List.range(0,K).cross(List.range(0,V)).map( z =>  {
          val dist = toDistribution( List.range(0,D).map( d => params_(z._1, d, z._2) ) )
          (z, dist)
        }).toMap

    DenseVector( Array.range(0,N).map(
      n => {
        val y = yDist.draw()
        val xs =  List.range[Int](0,V).map(v => xDist((y,v)).draw())
        DenseVector( (y::xs).toArray )
      }
    ))
  }

  def Z(params: Vector[Double]) = {
    1.0
  }

  /**
   * Compute the marginal likelihood of each observed state
   */
  def marginalLogLikelihood( params : Vector[Double], observed : Vector[Int] ) : Double = {
    val (full : DenseVector[Int]) = DenseVector.zeros[Int](variables.size)
    full.slice(1,full.length) := observed
    logsumexp(List.range(0,K).map(i => { full(0) = i; likelihood(params, full) } ))
  }

  /**
   * Compute posterior likelihood of hidden state values
   */
  def posterior( params : Vector[Double], observed : Vector[Int] ) : DenseVector[Double] = {
    val (full : DenseVector[Int]) = DenseVector.zeros[Int](variables.size)
    full.slice(1,full.length) := observed
    val (lhoods : List[Double]) =  List.range(0,K).map(i => { full(0) = i; likelihood(params, full) } )
    val norm = logsumexp(lhoods)
    DenseVector[Double]( lhoods.map( lhood => Math.exp(lhood - norm) ).toArray : Array[Double] )
  }


  def likelihood( params : Vector[Double], full : Vector[Int] ) = {
    // Compute the likelihood by adding the right parameters
    val yi = full(variables("y"))
    def yx_idx(i:Int) : String = s"y=$yi,x_$i="+ full(variables(s"x_$i"))
    List.range(0,V).foldLeft(params( features(s"y=$yi")))( (acc:Double,i:Int) => acc + params( features(yx_idx(i)) ) )
  }
  def marginal( params : Vector[Double], observed : Vector[Double] ) = DenseVector.zeros[Double](1)
}
object MixtureModel {
  def projectObserved(data: DenseVector[DenseVector[Int]]) : DenseVector[DenseVector[Int]] = {
    data.map( datum => datum.slice(1,datum.size) )
  }

  val rnd = new Random(1)

  def randomParameters( model : MixtureModel ) = DenseVector.rand(model.features.size, rnd)
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
    it( "should compute a valid posterior" ) {
      val model = new MixtureModel(2,2,3)
      val pr = model.posterior( DenseVector.zeros[Double](model.features.size), DenseVector.zeros[Int](model.variables.size-1) )
//      println(pr)
    }
    it( "should generate data" ) {
      val model = new MixtureModel(2,2,3)
      val params = MixtureModel.randomParameters(model)
      val N = 100
      val data = model.generate(params, N)
//      println(data)
    }
  }
}



