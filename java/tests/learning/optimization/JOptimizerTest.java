package learning.optimization;

import fig.basic.LogInfo;
import learning.linalg.FullTensor;
import learning.linalg.MatrixOps;
import learning.linalg.RandomFactory;
import learning.models.MixtureOfExperts;
import org.ejml.data.DenseMatrix64F;
import org.ejml.simple.SimpleMatrix;
import org.javatuples.Pair;
import org.junit.Assert;
import org.junit.Test;

import com.joptimizer.functions.*;
import com.joptimizer.optimizers.*;
import com.joptimizer.solvers.*;
import com.joptimizer.util.*;
import org.apache.commons.math3.linear.*;

import java.util.*;

public class JOptimizerTest {

  @Test
  public void lp() throws Exception {
		// Objective function (plane)
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(new double[] { -1., -1. }, 4);

		//inequalities (polyhedral feasible set G.X<H )
		ConvexMultivariateRealFunction[] inequalities = new ConvexMultivariateRealFunction[4];
		double[][] G = new double[][] {{4./3., -1}, {-1./2., 1.}, {-2., -1.}, {1./3., 1.}};
		double[] H = new double[] {2., 1./2., 2., 1./2.};
		inequalities[0] = new LinearMultivariateRealFunction(G[0], -H[0]);
		inequalities[1] = new LinearMultivariateRealFunction(G[1], -H[1]);
		inequalities[2] = new LinearMultivariateRealFunction(G[2], -H[2]);
		inequalities[3] = new LinearMultivariateRealFunction(G[3], -H[3]);
		
		//optimization problem
		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		or.setFi(inequalities);
		//or.setInitialPoint(new double[] {0.0, 0.0});//initial feasible point, not mandatory
		or.setToleranceFeas(1.E-9);
		or.setTolerance(1.E-9);
		
		//optimization
		JOptimizer opt = new JOptimizer();
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();

    double[] sol = opt.getOptimizationResponse().getSolution();
    Assert.assertTrue( MatrixOps.equal( sol[0], 1.5, 1e-2 ) );
    Assert.assertTrue( MatrixOps.equal( sol[1], 0.0, 1e-2 ) );
  }

  @Test
  public void sdp() throws Exception {
	// Objective function (variables (x,y,t), dim = 3)
		double[] c = new double[]{0,0,1};
		LinearMultivariateRealFunction objectiveFunction = new LinearMultivariateRealFunction(c, 0);
		
		//constraint in the form (A0.x+b0)T.(A0.x+b0) - c0.x - d0 - t < 0
		double[][] A0 = new double[][]{{-Math.sqrt(21./50.),  0.             , 0},
				                       {-Math.sqrt(2)/5.   , -1./Math.sqrt(2), 0}};
		double[] b0 = new double[] { 0, 0, 0 };
		double[] c0 = new double[] { 0, 0, 1 };
		double d0 = 0;
		
		//constraint (this is a circle) in the form (A1.x+b1)T.(A1.x+b1) - c1.x - d1 < 0
		double[][] A1 = new double[][]{{1,0,0},
				                       {0,1,0}};
		double[] b1 = new double[] { 2, 2, 0 };
		double[] c1 = new double[] { 0, 0, 0 };
		double d1 = Math.pow(1.75, 2);
		
		//matrix G for SDP
		double[][] G = new double[][]{{1     ,0     ,b0[0] ,0     ,0     ,0},
				                       {0     ,1     ,b0[1] ,0     ,0     ,0},
				                       {b0[0] ,b0[1] ,d0    ,0     ,0     ,0},
				                       {0     ,0     ,0     ,1     ,0     ,b1[0]},
				                       {0     ,0     ,0     ,0     ,1     ,b1[1]},
				                       {0     ,0     ,0     ,b1[0] ,b1[1] ,d1}};
		//matrices Fi for SDP
		double[][] F1 =  new double[][]{{0        ,0        ,A0[0][0] ,0        ,0        ,0},
						                {0        ,0        ,A0[1][0] ,0        ,0        ,0},
						                {A0[0][0] ,A0[1][0] ,c0[0]    ,0        ,0        ,0},
						                {0        ,0        ,0        ,0        ,0        ,A1[0][0]},
						                {0        ,0        ,0        ,0        ,0        ,A1[1][0]},
						                {0        ,0        ,0        ,A1[0][0] ,A1[1][0] ,c1[0]}};
		double[][] F2 =  new double[][]{{0        ,0        ,A0[0][1] ,0        ,0        ,0},
						                {0        ,0        ,A0[1][1] ,0        ,0        ,0},
						                {A0[0][1] ,A0[1][1] ,c0[1]    ,0        ,0        ,0},
						                {0        ,0        ,0        ,0        ,0        ,A1[0][1]},
						                {0        ,0        ,0        ,0        ,0        ,A1[1][1]},
						                {0        ,0        ,0        ,A1[0][1] ,A1[1][1] ,c1[1]}};
		double[][] F3 =  new double[][]{{0        ,0        ,A0[0][2] ,0        ,0        ,0},
						                {0        ,0        ,A0[1][2] ,0        ,0        ,0},
						                {A0[0][2] ,A0[1][2] ,c0[2]    ,0        ,0        ,0},
						                {0        ,0        ,0        ,0        ,0        ,A1[0][2]},
						                {0        ,0        ,0        ,0        ,0        ,A1[1][2]},
						                {0        ,0        ,0        ,A1[0][2] ,A1[1][2] ,c1[2]}};
		
		double[][] GMatrix = new Array2DRowRealMatrix(G).scalarMultiply(-1).getData();
		List<double[][]> FiMatrixList = new ArrayList<double[][]>();
		FiMatrixList.add(FiMatrixList.size(), new Array2DRowRealMatrix(F1).scalarMultiply(-1).getData());
		FiMatrixList.add(FiMatrixList.size(), new Array2DRowRealMatrix(F2).scalarMultiply(-1).getData());
		FiMatrixList.add(FiMatrixList.size(), new Array2DRowRealMatrix(F3).scalarMultiply(-1).getData());
		
		//optimization request
		OptimizationRequest or = new OptimizationRequest();
		or.setF0(objectiveFunction);
		//or.setInitialPoint(new double[] { -0.8, -0.8, 10});
		
		//optimization
		BarrierFunction bf = new SDPLogarithmicBarrier(FiMatrixList, GMatrix);
		BarrierMethod opt = new BarrierMethod(bf);
		opt.setOptimizationRequest(or);
		int returnCode = opt.optimize();

    double[] sol = opt.getOptimizationResponse().getSolution();
    Assert.assertTrue( MatrixOps.equal( sol[0], -2 + 1.75/Math.sqrt(2), 1e-2 ) );
    Assert.assertTrue( MatrixOps.equal( sol[1], -2 + 1.75/Math.sqrt(2), 1e-2 ) );
    Assert.assertTrue( MatrixOps.equal( sol[2],  0.8141035444, 1e-2 ) );
  }

  
}

