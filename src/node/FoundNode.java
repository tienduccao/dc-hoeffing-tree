package node;

/**
 * Wrapper of a leaf we found when filtering an instance to corresponding leaf
 */
public class FoundNode {
    public Node leafNode;

    public SplitNode parent;

    public int parentBranch;

    public FoundNode(Node leafNode, SplitNode parent, int parentBranch) {
        this.leafNode = leafNode;
        this.parent = parent;
        this.parentBranch = parentBranch;
    }
}
