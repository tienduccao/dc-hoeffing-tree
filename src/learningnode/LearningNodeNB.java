package learningnode;

import moa.classifiers.bayes.NaiveBayes;
import tree.DCHoeffdingTree;
import weka.core.Instance;

/**
 * Created by duccao on 08/02/16.
 */
public class LearningNodeNB extends ActiveLearningNode {

    private static final long serialVersionUID = 1L;

    public LearningNodeNB(double[] initialClassObservations) {
        super(initialClassObservations);
    }

    @Override
    public double[] getClassVotes(Instance inst, DCHoeffdingTree ht) {
        if (getWeightSeen() >= ht.nbThresholdOption.getValue()) {
            return NaiveBayes.doNaiveBayesPrediction(inst,
                    this.observedClassDistribution,
                    this.attributeObservers);
        }
        return super.getClassVotes(inst, ht);
    }

    @Override
    public void disableAttribute(int attIndex) {
        // should not disable poor atts - they are used in NB calc
    }
}