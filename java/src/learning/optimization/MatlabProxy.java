package learning.optimization;

import learning.Misc;
import learning.linalg.MatrixOps;
import learning.linalg.FullTensor;

import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;
import org.ejml.simple.SimpleMatrix;

import fig.basic.LogInfo;

import java.io.*;
import java.util.*;

import org.apache.commons.lang3.StringUtils;

/**
 * Runs Matlab as a proxy command, taking care of serialzing input and
 * output.
 */
public class MatlabProxy {
  public static void save( File path, SimpleMatrix X ) throws IOException {
    BufferedWriter writer = new BufferedWriter( new FileWriter( path ) );
    for( int r = 0; r < X.numRows(); r++ ) {
      for( int c = 0; c < X.numCols(); c++ ) {
        writer.write( String.format("%.16f  ", X.get(r,c) ) );
      }
      writer.newLine();
    }
    writer.newLine();
    writer.close();
  }
  public static void save( String path, SimpleMatrix X ) throws IOException {
    save( new File(path), X );
  }

  public static void save( File path, double x ) throws IOException {
    BufferedWriter writer = new BufferedWriter( new FileWriter( path ) );
    writer.write( String.format( "%.16f", x ) );
    writer.newLine();
    writer.close();
  }
  public static void save( String path, double x ) throws IOException {
    save( new File(path), x );
  }

  public static SimpleMatrix load( File path ) throws FileNotFoundException {
    ArrayList<double[]> data = new ArrayList<double[]>();
    Scanner input = new Scanner(path);
    while(input.hasNextLine())
    {
      Scanner rowReader = new Scanner(input.nextLine());
      ArrayList<Double> row = new ArrayList<Double>();
      while(rowReader.hasNextDouble())
      {
        row.add(rowReader.nextDouble());
      }
      data.add( Misc.unbox( row.toArray(new Double[] {}) ) );
    }
    return new SimpleMatrix( data.toArray(new double[][] {}) );
  }
  public static SimpleMatrix load( String path ) throws FileNotFoundException {
    return load( new File(path) );
  }

  public static SimpleMatrix run( File scriptPath, String cmd ) throws IOException, InterruptedException {
    LogInfo.begin_track("run-matlab");
    String matlabCmd = String.format( "addpath('%s'); %s; exit;", 
        scriptPath.getAbsolutePath(), 
        cmd
      );
    ProcessBuilder pb = new ProcessBuilder(
        "matlab",
        "-nojvm",
        "-nosplash",
        "-nodesktop",
        "-r",
        matlabCmd
        );
    pb.redirectErrorStream(true);
    LogInfo.logs(pb.command());
    Process proc = pb.start();

    InputStream is = proc.getInputStream();

    java.util.Scanner s = new java.util.Scanner(is).useDelimiter("\\A");
    LogInfo.logs( s.hasNext() ? s.next() : "" );

    proc.waitFor();
    
    LogInfo.end_track("run-matlab");
    return null;
  }
  public static SimpleMatrix run( String scriptPath, String cmd ) throws IOException, InterruptedException {
    return run( new File(scriptPath), cmd );
  }

  public static void main(String[] args) throws Exception {
    String path1 = "/tmp/data/";
    String path2 = System.getenv().get("HOME") + "/scr/spectral/matlab/";

    SimpleMatrix y = load( path2 + "data/y.txt" );
    SimpleMatrix X = load( path2 + "data/X.txt" );

    int N = X.numRows();
    int D = X.numCols();


    save( path1 + "y.txt", y );
    save( path1 + "X.txt", X );
    save( path1 + "lambda2.txt", 1e-3 );
    save( path1 + "lambda3.txt", 1e-3 );
    save( path1 + "sigma2.txt", 1e-1 );

    MatlabProxy.run( path2, String.format("sdpB2('%s')", path1) );
    MatlabProxy.run( path2, String.format("sdpB3('%s')", path1) );
    SimpleMatrix B2 = load( path1 + "B2.txt" );
    LogInfo.logs( B2 );

    SimpleMatrix B3_ = load( path1 + "B3.txt" );
    FullTensor B3 = FullTensor.reshape( B3_, new int[]{ D, D, D } );
    LogInfo.logs( B3 );
  }
}

