import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import weka.core.Instance;

/**
 * Created by duccao on 07/01/16.
 */
public class DCHoeffdingTree extends AbstractClassifier {
    /************************************************************
     *** Methods from AbstractClassifier
     ************************************************************/
    @Override
    public void resetLearningImpl() {
        // create the root node
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
        // find the leaf node and update sufficient stats

        // check whether we have to split
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    @Override
    public void getModelDescription(StringBuilder stringBuilder, int i) {

    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        return new double[0];
    }

    /************************************************************
     *** Implementation of HoeffdingTree
     ************************************************************/
}
