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

    protected AutoExpandVector<Node> children;

    public SplitNode(InstanceConditionalTest splitTest,
                     double[] classObservations, int size) {
        super(classObservations);
        this.splitTest = splitTest;
        this.children = new AutoExpandVector<>(size);
    }

    public void setChild(int index, Node child) {
        this.children.set(index, child);
    }

    public Node getChild(int index) {
        return this.children.get(index);
    }

    @Override
    public FoundNode filterInstanceToLeaf(Instance inst, SplitNode parent, int parentBranch) {
        int childIndex = this.splitTest.branchForInstance(inst);
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
