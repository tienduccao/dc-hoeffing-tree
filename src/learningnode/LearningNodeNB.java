package learningnode;

import moa.classifiers.bayes.NaiveBayes;
import tree.DCHoeffdingTree;
import weka.core.Instance;

/**
 * // TODO update doc of Naive Bayes option
 * LearningNode with Naive Bayes method
 */
public class LearningNodeNB extends ActiveLearningNode {

    private static final long serialVersionUID = 1L;

    public LearningNodeNB(double[] initialClassObservations) {
        super(initialClassObservations);
    }

    @Override
    public double[] getClassVotes(Instance instance, DCHoeffdingTree ht) {
        if (getWeightSeen() >= ht.nbThresholdOption.getValue()) {
            return NaiveBayes.doNaiveBayesPrediction(
                    instance,
                    this.observedClassDistribution,
                    this.attributeObservers
            );
        }
        return super.getClassVotes(instance, ht);
    }
}