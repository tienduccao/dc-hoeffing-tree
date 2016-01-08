import weka.core.Attribute;

/**
 * Created by duccao on 08/01/16.
 */
public class InformationGain implements SplitFunction {
    @Override
    public double value(Node node, Attribute attribute) {
        return 0;
    }

    /**
     * @param node
     * @return Entropy of a node
     */
    private double entropy(Node node) {
        int totalClassCount = 0;
        for (int numOfEachClass : node.getClassCounts()) {
            totalClassCount += numOfEachClass;
        }

        double entropy = 0;
        double probability;
        for (int numOfEachClass : node.getClassCounts()) {
            probability = (double) numOfEachClass / totalClassCount;
            entropy -= probability * Math.log(probability) / Math.log(2);
        }

        return entropy;
    }
}
