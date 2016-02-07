import weka.core.Attribute;

/**
 * Created by duccao on 08/01/16.
 */
public class InformationGain implements SplitFunction {
    @Override
    public double value(Node node, Attribute attribute) {
        return entropy(node, attribute);
    }

    /**
     * @param node
     * @param attribute
     * @return Entropy of a given node and attribute
     */
    private double entropy(Node node, Attribute attribute) {
        int totalCount = node.getTotalCountOfAttribute(attribute);

        double entropy = 0;
        double probabilityOfEachClass;
        for (int numOfEachClass : node.getClassCounts()) {
            probabilityOfEachClass = (double) numOfEachClass / totalCount;
            entropy -= probabilityOfEachClass * Math.log(probabilityOfEachClass) / Math.log(2);
        }

        return entropy;
    }
}
