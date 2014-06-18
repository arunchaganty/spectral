package learning.experiments;

import fig.basic.Indexer;
import fig.basic.LogInfo;
import fig.basic.Option;
import fig.exec.Execution;
import learning.common.Counter;
import learning.linalg.MatrixFactory;
import learning.linalg.MatrixOps;
import learning.models.HiddenMarkovModel;
import learning.models.Params;
import learning.models.loglinear.BinaryFeature;
import learning.models.loglinear.Example;
import learning.models.loglinear.Feature;
import learning.models.loglinear.Models;
import learning.unsupervised.ExpectationMaximization;
import org.ejml.simple.SimpleMatrix;

import java.util.Random;

/**
 * Does non-identifiability hurt?
 */
public class Identifiability implements  Runnable {

    @Option
    Random trueRandom = new Random(2);
    @Option
    double trueNoise = 1.0;
    @Option
    Random initRandom = new Random(3);
    @Option
    double initNoise = 1.0;

    SimpleMatrix getT(Params params) {
        Indexer<Feature> featureIndexer = params.getFeatureIndexer();
        SimpleMatrix mat = new SimpleMatrix(3,3);
        for(int h = 0; h < 3; h++)
            for(int h_ = 0; h_ < 3; h_++)
                mat.set(h,h_, params.get(new BinaryFeature(h, h_)));
        return mat;
    }
    void setT(Params params, SimpleMatrix mat) {
        Indexer<Feature> featureIndexer = params.getFeatureIndexer();
        for(int h = 0; h < 3; h++)
            for(int h_ = 0; h_ < 3; h_++)
                params.set(new BinaryFeature(h, h_), mat.get(h,h_));
    }


    public void run() {
        LogInfo.begin_track("Creating model");
        Models.HiddenMarkovModel model = new Models.HiddenMarkovModel(3, 3, 3);
        Params params = model.newParams();
        params.initRandom(trueRandom, trueNoise);

        SimpleMatrix T = getT(params);
        LogInfo.log(T);
        LogInfo.log(MatrixOps.svd(T).getValue1());
        LogInfo.end_track("Creating model");

        Counter<Example> data;
        ExpectationMaximization algo = new ExpectationMaximization();
        LogInfo.begin_track("Attempt recovery");
        {
            data = model.getFullDistribution(params);
            Params init = model.newParams();
            init.initRandom(initRandom, initNoise);
            init = algo.solveEM(model, data, init);

            double errors = 0.;
            for(Example ex : data) {
                Example ex_ = model.bestLabelling(init, ex);
                errors += (MatrixOps.equal(ex.h, ex_.h)) ? 0. : data.getCount(ex);
            }
            LogInfo.log("Error: " + errors);
        }

        LogInfo.end_track("Attempt recovery");


        LogInfo.begin_track("Making model un-identifiable");
        T = MatrixOps.approxk(T, 2);
        LogInfo.log(T);
        LogInfo.log(MatrixOps.svd(T).getValue1());
        LogInfo.end_track("Making model un-identifiable");
        setT(params, T);

        LogInfo.begin_track("Attempt recovery");
        {
            data = model.getDistribution(params);
            Params init = model.newParams();
            init.initRandom(initRandom, initNoise);
            init = algo.solveEM(model, data, init);

            double errors = 0.;
            for(Example ex : data) {
                Example ex_ = model.bestLabelling(init, ex);
                errors += (MatrixOps.equal(ex.h, ex_.h)) ? 0. : data.getCount(ex);
            }
            LogInfo.log("Error: " + errors);
        }

        LogInfo.end_track("Attempt recovery");
    }

    public static void main(String[] args) {
        Execution.run(args, new Identifiability());
    }
}
