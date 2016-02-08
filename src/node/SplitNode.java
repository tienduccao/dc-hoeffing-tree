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

    /**
     * @param instance
     * @param parent
     * @param parentBranch
     * @return The leaf corresponding to the given instance, split node and parent branch
     */
    @Override
    public FoundNode filterInstanceToLeaf(Instance instance, SplitNode parent, int parentBranch) {
        // find the branch of this child node
        int branch = this.splitTest.branchForInstance(instance);
        if (branch >= 0) {
            Node child = getChild(branch);
            if (child != null) {
                return child.filterInstanceToLeaf(instance, this, branch);
            }

            return new FoundNode(null, this, branch);
        }

        return new FoundNode(this, parent, parentBranch);
    }
}
