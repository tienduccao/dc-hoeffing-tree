package learningnode;

import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NullAttributeClassObserver;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.AutoExpandVector;
import moa.core.SizeOf;
import tree.DCHoeffdingTree;
import weka.core.Instance;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by duccao on 08/02/16.
 */
public class ActiveLearningNode extends LearningNode {

    private static final long serialVersionUID = 1L;

    protected double weightSeenAtLastSplitEvaluation;

    protected AutoExpandVector<AttributeClassObserver> attributeObservers = new AutoExpandVector<AttributeClassObserver>();

    protected boolean isInitialized;

    public ActiveLearningNode(double[] initialClassObservations) {
        super(initialClassObservations);
        this.weightSeenAtLastSplitEvaluation = getWeightSeen();
        this.isInitialized = false;
    }

    @Override
    public int calcByteSize() {
        return super.calcByteSize()
                + (int) (SizeOf.fullSizeOf(this.attributeObservers));
    }

    @Override
    public void learnFromInstance(Instance inst, DCHoeffdingTree ht) {
        if (this.isInitialized == false) {
            this.attributeObservers = new AutoExpandVector<AttributeClassObserver>(inst.numAttributes());
            this.isInitialized = true;
        }
        this.observedClassDistribution.addToValue((int) inst.classValue(),
                inst.weight());
        for (int i = 0; i < inst.numAttributes() - 1; i++) {
            int instAttIndex = inst.classIndex() > i? i: i + 1;;
            AttributeClassObserver obs = this.attributeObservers.get(i);
            if (obs == null) {
                obs = inst.attribute(instAttIndex).isNominal() ? ht.newNominalClassObserver() : ht.newNumericClassObserver();
                this.attributeObservers.set(i, obs);
            }
            obs.observeAttributeClass(inst.value(instAttIndex), (int) inst.classValue(), inst.weight());
        }
    }

    public double getWeightSeen() {
        return this.observedClassDistribution.sumOfValues();
    }

    public double getWeightSeenAtLastSplitEvaluation() {
        return this.weightSeenAtLastSplitEvaluation;
    }

    public void setWeightSeenAtLastSplitEvaluation(double weight) {
        this.weightSeenAtLastSplitEvaluation = weight;
    }

    public AttributeSplitSuggestion[] getBestSplitSuggestions(
            SplitCriterion criterion, DCHoeffdingTree ht) {
        List<AttributeSplitSuggestion> bestSuggestions = new LinkedList<AttributeSplitSuggestion>();
        double[] preSplitDist = this.observedClassDistribution.getArrayCopy();
        if (!ht.noPrePruneOption.isSet()) {
            // add null split as an option
            bestSuggestions.add(new AttributeSplitSuggestion(null,
                    new double[0][], criterion.getMeritOfSplit(
                    preSplitDist,
                    new double[][]{preSplitDist})));
        }
        for (int i = 0; i < this.attributeObservers.size(); i++) {
            AttributeClassObserver obs = this.attributeObservers.get(i);
            if (obs != null) {
                AttributeSplitSuggestion bestSuggestion = obs.getBestEvaluatedSplitSuggestion(criterion,
                        preSplitDist, i, ht.binarySplitsOption.isSet());
                if (bestSuggestion != null) {
                    bestSuggestions.add(bestSuggestion);
                }
            }
        }
        return bestSuggestions.toArray(new AttributeSplitSuggestion[bestSuggestions.size()]);
    }

    public void disableAttribute(int attIndex) {
        this.attributeObservers.set(attIndex,
                new NullAttributeClassObserver());
    }
}