package node;

/**
 * Created by duccao on 08/02/16.
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
