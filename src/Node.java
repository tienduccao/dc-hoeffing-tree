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

    /** Total number of instances in this node */
    private int numOfInstances;

    /** Total number of instances of the majority class */
    private int numOfMajorityClassInstances;

    /** Store number of each class */
    private int classCounts[];

    /** Sufficient stats [attribute index][attribute value][class] */
    private int sufficientStats[][][];

    /** Class value if this node is a leaf */
    private double classValue;

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
     * Update this node when new instance arrived
     * @param instance
     */
    public void update(Instance instance) {
        // increase total number of instances
        numOfInstances++;

        // increase count for corresponding class
        // assume that we only have nominal class
        int incomingClassValue = (int) instance.classValue();
        classCounts[incomingClassValue]++;

        // update the majority class
        if (classCounts[incomingClassValue] > numOfMajorityClassInstances) {
            numOfMajorityClassInstances = classCounts[incomingClassValue];
            classValue = instance.value(classAttribute);
        }

        // update the sufficient stats
        Attribute attribute;
        int attributeIndex, attributeValue;
        for (int i = 0; i < instance.numAttributes(); ++i) {
            attribute = instance.attribute(i);

            attributeIndex = attribute.index();
            attributeValue = (int) instance.value(attribute);

            sufficientStats[attributeIndex][attributeValue][(int) classValue]++;
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
