package learningnode;

import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.AutoExpandVector;
import tree.DCHoeffdingTree;
import weka.core.Instance;

import java.util.LinkedList;
import java.util.List;

/**
 * LearningNode with Majority Class approach
 */
public class ActiveLearningNode extends LearningNode {

    private static final long serialVersionUID = 1L;

    protected double weightSeenAtLastSplitEvaluation;

    // for observing the class data distribution for each attribute
    protected AutoExpandVector<AttributeClassObserver> attributeObservers = new AutoExpandVector<>();

    protected boolean isInitialized;

    public ActiveLearningNode(double[] initialClassObservations) {
        super(initialClassObservations);
        this.weightSeenAtLastSplitEvaluation = getWeightSeen();
        this.isInitialized = false;
    }

    @Override
    public void learnFromInstance(Instance instance, DCHoeffdingTree ht) {
        if (!this.isInitialized) {
            this.attributeObservers = new AutoExpandVector<>(instance.numAttributes());
            this.isInitialized = true;
        }

        // increase weight of this class value
        this.observedClassDistribution.addToValue((int) instance.classValue(), instance.weight());

        for (int i = 0; i < instance.numAttributes() - 1; i++) {
            // skip the class attribute
            int attributeIndex = instance.classIndex() > i ? i : i + 1;

            AttributeClassObserver attributeClassObserver = this.attributeObservers.get(i);
            if (attributeClassObserver == null) {
                attributeClassObserver = instance.attribute(attributeIndex).isNominal() ?
                        ht.newNominalClassObserver() : ht.newNumericClassObserver();
                this.attributeObservers.set(i, attributeClassObserver);
            }

            // update statistics of the given attribute
            attributeClassObserver.observeAttributeClass(
                    instance.value(attributeIndex),
                    (int) instance.classValue(),
                    instance.weight()
            );
        }
    }

    public AttributeSplitSuggestion[] getBestSplitSuggestions(SplitCriterion criterion, DCHoeffdingTree ht) {
        List<AttributeSplitSuggestion> bestSuggestions = new LinkedList<>();
        double[] preSplitDistribution = this.observedClassDistribution.getArrayCopy();

        for (int attributeIndex = 0; attributeIndex < this.attributeObservers.size(); attributeIndex++) {
            AttributeClassObserver attributeClassObserver = this.attributeObservers.get(attributeIndex);
            if (attributeClassObserver != null) {
                AttributeSplitSuggestion bestSuggestion = attributeClassObserver.getBestEvaluatedSplitSuggestion(
                        criterion,
                        preSplitDistribution,
                        attributeIndex,
                        ht.binarySplitsOption.isSet()
                );
                if (bestSuggestion != null) {
                    bestSuggestions.add(bestSuggestion);
                }
            }
        }

        return bestSuggestions.toArray(new AttributeSplitSuggestion[bestSuggestions.size()]);
    }

    /*************************/
    /** Getters and setters **/
    /*************************/

    /**
     * @return sum of weights from all class values
     */
    public double getWeightSeen() {
        return this.observedClassDistribution.sumOfValues();
    }

    public double getWeightSeenAtLastSplitEvaluation() {
        return this.weightSeenAtLastSplitEvaluation;
    }

    public void setWeightSeenAtLastSplitEvaluation(double weight) {
        this.weightSeenAtLastSplitEvaluation = weight;
    }
}