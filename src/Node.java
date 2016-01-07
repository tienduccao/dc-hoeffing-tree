import weka.core.Attribute;

/**
 * Created by duccao on 07/01/16.
 */
public class Node {
    private Attribute classAttribute;

    public Node(Attribute[] attributes, Attribute classAttribute) {
        this.classAttribute = classAttribute;
    }
}
