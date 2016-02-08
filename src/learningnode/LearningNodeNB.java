package learningnode;

import moa.classifiers.bayes.NaiveBayes;
import tree.DCHoeffdingTree;
import weka.core.Instance;

/**
 * LearningNode which uses Naive Bayes for predicting class of leaf
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