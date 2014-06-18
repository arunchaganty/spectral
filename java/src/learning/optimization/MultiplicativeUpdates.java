package learning.optimization;

import fig.basic.Maximizer;
import learning.linalg.MatrixOps;

/**
* Created by chaganty on 1/23/14.
*/
public class MultiplicativeUpdates extends Maximizer {
  double stepSize;
  int steps = 0;

  public MultiplicativeUpdates(double stepSize) {
    this.stepSize = stepSize;
  }
  public MultiplicativeUpdates() {
    this(1.0);
  }

  @Override
  public boolean takeStep(FunctionState func) {
    // w_{t+1} \propto w_t e^{\eta z_t}
    double[] point = func.point();
    double[] gradient = func.gradient();

    for(int i = 0; i < point.length; i++) {
      point[i] *= Math.exp(stepSize * gradient[i]);
    }
    double Z = MatrixOps.sum(point);
    MatrixOps.scale(point, 1./Z);

    // Anneal the step size. (1/sqrt(1+t))
    stepSize = stepSize * Math.sqrt((1+steps)/(2+steps));
    steps++;
    return false;
  }
}
