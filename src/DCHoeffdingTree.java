import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import weka.core.Attribute;
import weka.core.Instance;

/**
 * Created by duccao on 07/01/16.
 */
public class DCHoeffdingTree extends AbstractClassifier {
    // Configurations
    private double delta = 0.05;
    private int nMin = 30;
    private SplitFunction splitFunction = new InformationGain();

    // Variables
    private Node root;
    double R;   // for calculating the Hoeffding bound

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
        Node node = root.findLeafNode(instance);
        node.update(instance);

        // check whether we have to split
        if (node.getNumOfInstances() % nMin == 0) {
            attemptToSplit(node, instance);
        }
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
        double[] votes = new double[instance.numClasses()];
        votes[(int) root.findLeafNode(instance).getClassValue()] = 1.0;
        return votes;
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

    private void attemptToSplit(Node node, Instance instance) {
        int classAttributeIndex = instance.classAttribute().index();

        // find 2 attributes with highest values from split function
        double firstValue = Double.MIN_VALUE;
        double secondValue = Double.MIN_VALUE;
        Attribute attributeWithHighestValue = null;
        for (int attributeIndex = 0; attributeIndex < instance.numAttributes(); ++attributeIndex) {
            // skip the class attribute
            if (attributeIndex == classAttributeIndex) continue;

            Attribute attribute = instance.attribute(attributeIndex);
            double value = splitFunction.value(node, attribute);
            if (value > firstValue) {
                firstValue = value;
                attributeWithHighestValue = attribute;
            } else if (value > secondValue) {
                secondValue = value;
            }
        }

        double delta = firstValue - secondValue;

        // check to see whether we can split
        if (delta > calculateHoeffdingBound(node, instance)) {
            node.split(attributeWithHighestValue, instance);
        }
    }

    private double calculateHoeffdingBound(Node node, Instance instance) {
        int n = node.getNumOfInstances();
        // TODO log or log2
        R = Math.log(instance.numClasses());
        return Math.sqrt( Math.pow(R, 2) * Math.log(1 / delta) / (2 * n) );
    }
}
