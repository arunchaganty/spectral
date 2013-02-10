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
  if( args.length != 3 ) {
    println("Usage: N nlType <out-directory> ");
    System.exit(1);
  }
}

def generate( N:String, outputPath : File, d:Int, nld:Int, nlType:String, k:Int ){
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
    -('outputPath, outputPath.getAbsolutePath() + File.separator + d + "-" + nld + "-" + k + ".dat")
    -('D, d)
    -('nlDegree, nld)
    -('K, k)
  )}

def main() {
  checkArgs(args)

  val N = args(0);
  val nlType = args(1);
  val outputPath = new File( args(2) ++ File.separator ++ nlType);
  // Make output directory
  outputPath.mkdirs();

  // Construct a list of D, nD, K
  val D = List(1,2)
  val nlD = List(3,4,5)
  val K = List(2,3,5,7)
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
          k <= nl.getLinearDimension(d) 
      })
  // Spawn creation jobs
  for (( d, nd, k ) <- candidateTriples ) {
    println( d, nd, k );
    generate( N, outputPath, d, nd, nlType, k );
  }
}
main()
