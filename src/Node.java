import moa.AbstractMOAObject;
import moa.core.DoubleVector;
import moa.core.SizeOf;
import moa.core.StringUtils;
import weka.core.Instance;

/**
 * Created by duccao on 08/02/16.
 */
public class Node extends AbstractMOAObject {

    private static final long serialVersionUID = 1L;

    protected DoubleVector observedClassDistribution;

    public Node(double[] classObservations) {
        this.observedClassDistribution = new DoubleVector(classObservations);
    }

    public int calcByteSize() {
        return (int) (SizeOf.sizeOf(this) + SizeOf.fullSizeOf(this.observedClassDistribution));
    }

    public int calcByteSizeIncludingSubtree() {
        return calcByteSize();
    }

    public boolean isLeaf() {
        return true;
    }

    public FoundNode filterInstanceToLeaf(Instance inst, SplitNode parent,
                                          int parentBranch) {
        return new FoundNode(this, parent, parentBranch);
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

    public void describeSubtree(HoeffdingTree ht, StringBuilder out,
                                int indent) {
        StringUtils.appendIndented(out, indent, "Leaf ");
        out.append(ht.getClassNameString());
        out.append(" = ");
        out.append(ht.getClassLabelString(this.observedClassDistribution.maxIndex()));
        out.append(" weights: ");
        this.observedClassDistribution.getSingleLineDescription(out,
                ht.treeRoot.observedClassDistribution.numValues());
        StringUtils.appendNewline(out);
    }

    public int subtreeDepth() {
        return 0;
    }

    public double calculatePromise() {
        double totalSeen = this.observedClassDistribution.sumOfValues();
        return totalSeen > 0.0 ? (totalSeen - this.observedClassDistribution.getValue(this.observedClassDistribution.maxIndex()))
                : 0.0;
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
        describeSubtree(null, sb, indent);
    }
}
