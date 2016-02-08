package node;

import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
import moa.core.AutoExpandVector;
import moa.core.SizeOf;
import moa.core.StringUtils;
import tree.DCHoeffdingTree;
import weka.core.Instance;

/**
 * Created by duccao on 08/02/16.
 */
public class SplitNode extends Node {

    private static final long serialVersionUID = 1L;

    protected InstanceConditionalTest splitTest;

    protected AutoExpandVector<Node> children; // = new AutoExpandVector<Node>();

    @Override
    public int calcByteSize() {
        return super.calcByteSize()
                + (int) (SizeOf.sizeOf(this.children) + SizeOf.fullSizeOf(this.splitTest));
    }

    @Override
    public int calcByteSizeIncludingSubtree() {
        int byteSize = calcByteSize();
        for (Node child : this.children) {
            if (child != null) {
                byteSize += child.calcByteSizeIncludingSubtree();
            }
        }
        return byteSize;
    }

    public SplitNode(InstanceConditionalTest splitTest,
                     double[] classObservations, int size) {
        super(classObservations);
        this.splitTest = splitTest;
        this.children = new AutoExpandVector<Node>(size);
    }

    public SplitNode(InstanceConditionalTest splitTest,
                     double[] classObservations) {
        super(classObservations);
        this.splitTest = splitTest;
        this.children = new AutoExpandVector<Node>();
    }


    public int numChildren() {
        return this.children.size();
    }

    public void setChild(int index, Node child) {
        if ((this.splitTest.maxBranches() >= 0)
                && (index >= this.splitTest.maxBranches())) {
            throw new IndexOutOfBoundsException();
        }
        this.children.set(index, child);
    }

    public Node getChild(int index) {
        return this.children.get(index);
    }

    public int instanceChildIndex(Instance inst) {
        return this.splitTest.branchForInstance(inst);
    }

    @Override
    public boolean isLeaf() {
        return false;
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

    @Override
    public void describeSubtree(DCHoeffdingTree ht, StringBuilder out,
                                int indent) {
        for (int branch = 0; branch < numChildren(); branch++) {
            Node child = getChild(branch);
            if (child != null) {
                StringUtils.appendIndented(out, indent, "if ");
                out.append(this.splitTest.describeConditionForBranch(branch,
                        ht.getModelContext()));
                out.append(": ");
                StringUtils.appendNewline(out);
                child.describeSubtree(ht, out, indent + 2);
            }
        }
    }

    @Override
    public int subtreeDepth() {
        int maxChildDepth = 0;
        for (Node child : this.children) {
            if (child != null) {
                int depth = child.subtreeDepth();
                if (depth > maxChildDepth) {
                    maxChildDepth = depth;
                }
            }
        }
        return maxChildDepth + 1;
    }
}
