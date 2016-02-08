import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import weka.core.Instance;

/**
 * Created by duccao on 08/02/16.
 */
public class DCHoeffdingTree extends AbstractClassifier {
    /************************************************************
     *** Variables
     ************************************************************/
    protected Node treeRoot;

    protected int decisionNodeCount;

    protected int activeLeafNodeCount;

    protected int inactiveLeafNodeCount;

    /************************************************************
     *** Methods from AbstractClassifier
     ************************************************************/
    @Override
    public void resetLearningImpl() {

    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
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
     *** HoeffdingTree implementation
     ************************************************************/
}
