package node;

import moa.AbstractMOAObject;
import moa.core.DoubleVector;
import tree.DCHoeffdingTree;
import weka.core.Instance;

/**
 * Node of tree
 */
public class Node extends AbstractMOAObject {

    private static final long serialVersionUID = 1L;

    protected DoubleVector observedClassDistribution;

    public Node(double[] classObservations) {
        this.observedClassDistribution = new DoubleVector(classObservations);
    }

    public FoundNode filterInstanceToLeaf(Instance inst, SplitNode parent, int parentBranch) {
        return new FoundNode(this, parent, parentBranch);
    }

    public FoundNode filterInstanceToLeaf(Instance instance) {
        return this.filterInstanceToLeaf(instance, null, -1);
    }

    public double[] getObservedClassDistribution() {
        return this.observedClassDistribution.getArrayCopy();
    }

    public double[] getClassVotes(Instance inst, DCHoeffdingTree ht) {
        return this.observedClassDistribution.getArrayCopy();
    }

    public boolean observedClassDistributionIsPure() {
        return this.observedClassDistribution.numNonZeroEntries() < 2;
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
    }
}
