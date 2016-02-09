package learningnode;

import node.Node;
import tree.DCHoeffdingTree;
import weka.core.Instance;

/**
 * Base class for nodes which can learn from incoming instances
 */
public abstract class LearningNode extends Node {

    private static final long serialVersionUID = 1L;

    public LearningNode(double[] initialClassObservations) {
        super(initialClassObservations);
    }

    public abstract void learnFromInstance(Instance instance, DCHoeffdingTree ht);
}
