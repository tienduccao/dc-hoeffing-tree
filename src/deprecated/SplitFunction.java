package deprecated;

import weka.core.Attribute;

/**
 * Created by duccao on 08/01/16.
 */
public interface SplitFunction {
    double value(Node node, Attribute attribute);
}
