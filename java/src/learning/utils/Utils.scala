package learning.utils

/**
 * Various scala utilities
 */
object Utils {
  implicit class Crossable[X](xs: Traversable[X]) {
    def cross[Y](ys: Traversable[Y]) = for { x <- xs; y <- ys } yield (x, y)
    def cross3[Y,Z](ys: Traversable[Y], zs: Traversable[Z]) = for { x <- xs; y <- ys; z <- zs } yield (x, y, z)
  }

  def logsumexp(x : Double, y : Double) : Double =
    if ( x >  y )
       y + Math.log( 1 + Math.exp(x - y) )
    else
       x + Math.log( 1 + Math.exp(y - x) )

  def logsumexp( xs: Traversable[Double] ) : Double =
    xs.reduceLeft( (acc:Double, x:Double) => logsumexp(acc,x) )

}

