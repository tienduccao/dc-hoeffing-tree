import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import weka.core.Attribute;
import weka.core.Instance;

/**
 * Created by duccao on 07/01/16.
 */
public class DCHoeffdingTree extends AbstractClassifier {
    private Node root;

    /************************************************************
     *** Methods from AbstractClassifier
     ************************************************************/
    @Override
    public void resetLearningImpl() {
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
        // create the root node if necessary
        if (root == null) {
            root = createRootNode(instance);
        }

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
    private Node createRootNode(Instance instance) {
        int numAttributes = instance.numAttributes();
        Attribute[] attributes = new Attribute[numAttributes];

        for (int i = 0; i < numAttributes; ++i) {
            attributes[i] = instance.attribute(i);
        }

        return new Node(attributes, instance.classAttribute());
    }
}
