package node;

/**
 * Wrapper of a node we found when filtering an instance to leaf
 */
public class FoundNode {
    public Node node;

    public SplitNode parent;

    public int parentBranch;

    public FoundNode(Node node, SplitNode parent, int parentBranch) {
        this.node = node;
        this.parent = parent;
        this.parentBranch = parentBranch;
    }
}
