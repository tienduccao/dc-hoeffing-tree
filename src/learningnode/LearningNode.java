package learningnode;

import node.Node;
import tree.DCHoeffdingTree;
import weka.core.Instance;

/**
 * Created by duccao on 08/02/16.
 */
public abstract class LearningNode extends Node {

    private static final long serialVersionUID = 1L;

    public LearningNode(double[] initialClassObservations) {
        super(initialClassObservations);
    }

    public abstract void learnFromInstance(Instance inst, DCHoeffdingTree ht);
}
