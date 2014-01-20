package learning.utils;

import fig.basic.IOUtils;
import fig.basic.LogInfo;
import fig.basic.Maximizer;
import fig.basic.StrUtils;
import fig.exec.Execution;
import learning.data.ComputableMoments;
import learning.data.HasExactMoments;
import learning.linalg.FullTensor;
import learning.linalg.MatrixFactory;
import learning.linalg.MatrixOps;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Quartet;

import java.io.*;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

/**
 * Various utilities
 */
public class UtilsJ {
  public static ComputableMoments fromExactMoments(HasExactMoments obj) {
    final Quartet<SimpleMatrix, SimpleMatrix, SimpleMatrix, FullTensor> moments = obj.computeExactMoments();
    return new ComputableMoments() {
      @Override
      public MatrixOps.Matrixable computeP13() {
        return MatrixOps.matrixable( moments.getValue0() );
      }

      @Override
      public MatrixOps.Matrixable computeP12() {
        return MatrixOps.matrixable( moments.getValue1() );
      }

      @Override
      public MatrixOps.Matrixable computeP32() {
        return MatrixOps.matrixable( moments.getValue2() );
      }

      @Override
      public MatrixOps.Tensorable computeP123() {
        return MatrixOps.tensorable( moments.getValue3() );
      }
    };
  }

  public static void doGradientCheck(Maximizer.FunctionState state) {
    double epsilon = 1e-4;
    // Save point
    state.invalidate();
    double[] point = state.point();
//    double value = state.value();
    double[] gradient = state.gradient();
    double[] currentGradient = gradient.clone();
    double[] currentPoint = point.clone();


    // Set point to be +/- gradient
    for( int i = 0; i < currentPoint.length; i++ ) {
      point[i] = currentPoint[i] + epsilon;
      state.invalidate();
      double valuePlus = state.value();
      point[i] = currentPoint[i] - epsilon;
      state.invalidate();
      double valueMinus = state.value();
      point[i] = currentPoint[i];
      state.invalidate();

      double expectedValue = (valuePlus - valueMinus)/(2*epsilon);
      double actualValue = currentGradient[i];
      assert MatrixOps.equal(expectedValue, actualValue, 1e-4);
    }
  }

  public static String readFully(BufferedReader reader) throws IOException {
    StringBuilder sb = new StringBuilder();
    String line;
    while( (line = reader.readLine()) != null )
      sb.append(line).append("\n");
    return sb.toString().trim();
  }

  public static void writeString(String path, String str) throws IOException {
    BufferedWriter writer = Files.newBufferedWriter(new File(path).toPath(), Charset.defaultCharset());
    writer.write(str);
    writer.close();
  }
  public static void writeStringHard(String path, String str) {
    try {
      writeString(path, str);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static BufferedReader openReader(File file) throws FileNotFoundException {
    return new BufferedReader(new FileReader(file));
  }

  public static boolean optimize(Maximizer maximizer, Maximizer.FunctionState state, String label, int numIters, boolean diagnosticMode) {
    LogInfo.begin_track("optimize " + label);
//    state.invalidate();

    PrintWriter out = null;
    if(Execution.getActualExecDir() != null) {
      out = IOUtils.openOutHard(Execution.getFile(label + ".events"));
    }

    boolean done = false;
    double oldObjective = Double.NEGATIVE_INFINITY;
    int iter;
    for (iter = 0; iter < numIters && !done; iter++) {
      if(diagnosticMode) {
        doGradientCheck(state);
      }
      // Logging stuff
      List<String> items = new ArrayList<>();
      items.add("iter = " + iter);
      items.add("objective = " + state.value());
      items.add("pointNorm = " + MatrixOps.norm(state.point()));
      items.add("gradientNorm = " + MatrixOps.norm(state.gradient()));
      LogInfo.log( StrUtils.join(items, "\t") );
      if(out != null) {
        out.println( StrUtils.join(items, "\t") );
        out.flush();
      }

      double objective = state.value();
      assert objective > oldObjective;
      oldObjective = objective;

      done = maximizer.takeStep(state);
    }
    // Do a gradient check only at the very end.
    if(diagnosticMode) {
      doGradientCheck(state);
    }

    List<String> items = new ArrayList<>();
    items.add("iter = " + iter);
    items.add("objective = " + state.value());
    items.add("pointNorm = " + MatrixOps.norm(state.point()));
    items.add("gradientNorm = " + MatrixOps.norm(state.gradient()));
    LogInfo.log( StrUtils.join(items, "\t") );
    if(out!=null){
      out.println( StrUtils.join(items, "\t") );
      out.flush();
    }

    LogInfo.end_track();
    return done && iter == 1;  // Didn't make any updates
  }


  public static String outputList(Object... items) {
    // Each item is a pair
    assert items.length % 2 == 0;
    StringBuilder sb = new StringBuilder();
    for(int item = 0; item < items.length; item += 2) {
      sb.append(items[item]).append(" = ").append(items[item+1]).append("\t");
    }

    return sb.toString().trim();
  }

}
