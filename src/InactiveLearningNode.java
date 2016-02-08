import weka.core.Instance;

/**
 * Created by duccao on 08/02/16.
 */
public class InactiveLearningNode extends LearningNode {

    private static final long serialVersionUID = 1L;

    public InactiveLearningNode(double[] initialClassObservations) {
        super(initialClassObservations);
    }

    @Override
    public void learnFromInstance(Instance inst, DCHoeffdingTree ht) {
        this.observedClassDistribution.addToValue((int) inst.classValue(), inst.weight());
    }
}
