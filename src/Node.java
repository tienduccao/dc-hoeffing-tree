import weka.core.Attribute;
import weka.core.Instance;

/**
 * Created by duccao on 07/01/16.
 */
public class Node {
    /** Attribute to determine class of this node */
    private Attribute classAttribute;

    /** Attribute used to split this node */
    private Attribute splittingAttribute;

    /** Children of this node */
    private Node[] children;

    public Node(Attribute[] attributes, Attribute classAttribute) {
        this.classAttribute = classAttribute;
    }

    /**
     * @param instance
     * @return The leaf node associated with this given instance
     */
    public Node findLeafNode(Instance instance) {
        return findLeafNode(this, instance);
    }

    /**
     * @param node
     * @param instance
     * @return The leaf node associated with this given
     * (internal or root) node and instance
     */
    public Node findLeafNode(Node node, Instance instance) {
        if (node == null || node.isLeafNode()) {
            return node;
        } else {
            Node childNode = null;
            if (children != null) {
                int attributeValue = (int) instance.value(node.getSplittingAttribute());
                // the child nodes are created by associating each attribute value with a new child node
                childNode = children[attributeValue];
            }

            return findLeafNode(childNode, instance);
        }
    }

    /**
     * @return true if this node is a leaf
     */
    public boolean isLeafNode() {
        return this.children == null;
    }

    public Attribute getSplittingAttribute() {
        return splittingAttribute;
    }
}
