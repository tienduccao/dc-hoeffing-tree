package node;

import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.core.AutoExpandVector;
import weka.core.Instance;

/**
 * Created by duccao on 08/02/16.
 */
public class SplitNode extends Node {

    private static final long serialVersionUID = 1L;

    protected InstanceConditionalTest splitTest;

    protected AutoExpandVector<Node> children; // = new AutoExpandVector<Node>();

    public SplitNode(InstanceConditionalTest splitTest,
                     double[] classObservations, int size) {
        super(classObservations);
        this.splitTest = splitTest;
        this.children = new AutoExpandVector<Node>(size);
    }

    public void setChild(int index, Node child) {
        this.children.set(index, child);
    }

    public Node getChild(int index) {
        return this.children.get(index);
    }

    public int instanceChildIndex(Instance inst) {
        return this.splitTest.branchForInstance(inst);
    }

    @Override
    public FoundNode filterInstanceToLeaf(Instance inst, SplitNode parent,
                                          int parentBranch) {
        int childIndex = instanceChildIndex(inst);
        if (childIndex >= 0) {
            Node child = getChild(childIndex);
            if (child != null) {
                return child.filterInstanceToLeaf(inst, this, childIndex);
            }
            return new FoundNode(null, this, childIndex);
        }
        return new FoundNode(this, parent, parentBranch);
    }
}
