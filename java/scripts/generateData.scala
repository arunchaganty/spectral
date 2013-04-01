/*
 * This script generates data in a way that ensures valid
 * dimensionality, etc.
 */
import learning.models.transforms._
import java.io.File;
import Qry._

val CLASSPATH = "bin:" + (new File("deps/")).listFiles().mkString( ":" )

// Check arguments
def checkArgs(args:Array[String]) {
  if( args.length != 4 ) {
    println("Usage: N E nlType <out-directory> ");
    System.exit(1);
  }
}

def generate( N:String, outputPath : File, d:Int, nld:Int, nlType:String, k:Int, e:Int ){
  submit("java"
    -('cp, CLASSPATH)
    -"Xmx2g"
    -"Xms2g"
    -"XX:MaxPermSize=64M"
    -'ea
    ->"learning.models.MixtureOfExperts"
    //
    // Global properties file for the KBP project
    //
    -('N, N)
    -('betas, "random")
    -('sigma2, 0.1)
    -('bias, false)
    -('nlType, nlType)
    -('outputPath, outputPath.getAbsolutePath() + File.separator + d + "-" + nld + "-" + k + "." + e + ".dat")
    -('D, d)
    -('nlDegree, nld)
    -('K, k)
  )}

def main() {
  checkArgs(args)

  val N = args(0);
  val E = args(1).toInt;
  val nlType = args(2);
  val outputPath = new File( args(3) ++ File.separator ++ nlType);
  // Make output directory
  outputPath.mkdirs();

  // Construct a list of D, nD, K
  val D = List(20, 25, 30)
  val nlD = List(1) // List(3,4,5)
  val K = List(5,9)
  val candidateTriples = 
    (for (d <- D; nld <- nlD; k <- K) yield (d, nld, k))
    .filter( x  => 
      x match {
        case ( d, nd, k ) =>
          // Check if this triple satisfies conditions
          val nl = 
            nlType match {
              case "poly-independent" => new IndependentPolynomial(nd, d)
              case "poly" => new PolynomialNonLinearity(nd)
              case "random-fractional" => new FractionalPolynomial(nd, d)
              case "fourier" => new FourierNonLinearity(nd)
            }
          k <= nl.getLinearDimension(d) //&& nl.getLinearDimension(d) <= 20
      })
  // Spawn creation jobs
  for (( d, nd, k ) <- candidateTriples ) {
    println( d, nd, k );
    for( e <- 1 to E )
      generate( N, outputPath, d, nd, nlType, k, e );
  }
}
main()
