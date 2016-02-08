package tree;

import learningnode.ActiveLearningNode;
import learningnode.LearningNode;
import learningnode.LearningNodeNB;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.DiscreteAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NumericAttributeClassObserver;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.Measurement;
import moa.options.ClassOption;
import moa.options.FlagOption;
import moa.options.FloatOption;
import moa.options.IntOption;
import node.FoundNode;
import node.Node;
import node.SplitNode;
import weka.core.Instance;

import java.util.Arrays;

/**
 * Created by duccao on 08/02/16.
 */
public class DCHoeffdingTree extends AbstractClassifier {
    public IntOption nMinOption = new IntOption(
            "nMin",
            'g',
            "The number of instances a leaf should observe between split attempts.",
            200, 0, Integer.MAX_VALUE);

    public IntOption nbThresholdOption = new IntOption(
            "nbThreshold",
            'q',
            "The number of instances a leaf should observe before permitting Naive Bayes.",
            0, 0, Integer.MAX_VALUE);

    public ClassOption numericEstimatorOption = new ClassOption(
            "numericEstimator",
            'n', "Numeric estimator to use.",
            NumericAttributeClassObserver.class,
            "GaussianNumericAttributeClassObserver");

    public ClassOption nominalEstimatorOption = new ClassOption(
            "nominalEstimator",
            'd',
            "Nominal estimator to use.",
            DiscreteAttributeClassObserver.class,
            "NominalAttributeClassObserver");

    public ClassOption splitCriterionOption = new ClassOption(
            "splitCriterion",
            's',
            "Split criterion to use.", 
            SplitCriterion.class,
            "InfoGainSplitCriterion");

    public FloatOption splitConfidenceOption = new FloatOption(
            "splitConfidence",
            'c',
            "The allowable error in split decision, values closer to 0 will take longer to decide.",
            0.0000001, 0.0, 1.0);

    public FloatOption tieThresholdOption = new FloatOption("tieThreshold",
            't', "Threshold below which a split will be forced to break ties.",
            0.05, 0.0, 1.0);

    public FlagOption binarySplitsOption = new FlagOption(
            "binarySplits",
            'b',
            "Only allow binary splits.");

    /************************************************************
     *** Variables
     ************************************************************/
    protected Node treeRoot;

    /************************************************************
     *** Methods from AbstractClassifier
     ************************************************************/
    @Override
    public void resetLearningImpl() {
        this.treeRoot = null;
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {
        // Initialize the root (if necessary)
        if (this.treeRoot == null) {
            this.treeRoot = newLearningNode();
        }

        // find the leaf associated with this instance
        FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(instance, null, -1);
        Node leafNode = foundNode.node;
        if (leafNode == null) {
            leafNode = newLearningNode();
            foundNode.parent.setChild(foundNode.parentBranch, leafNode);
        }

        if (leafNode instanceof ActiveLearningNode) {
            ActiveLearningNode activeLearningNode = (ActiveLearningNode) leafNode;
            activeLearningNode.learnFromInstance(instance, this);

            // observe at least nMin examples before trying to split
            double weightSeen = activeLearningNode.getWeightSeen();
            if (weightSeen - activeLearningNode.getWeightSeenAtLastSplitEvaluation()
                    >= this.nMinOption.getValue()) {
                attemptToSplit(activeLearningNode, foundNode.parent, foundNode.parentBranch);
                activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
            }
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[0];
    }

    @Override
    public void getModelDescription(StringBuilder stringBuilder, int i) {

    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        if (this.treeRoot != null) {
            FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst, null, -1);
            Node leafNode = foundNode.node;
            if (leafNode == null) {
                leafNode = foundNode.parent;
            }
            return leafNode.getClassVotes(inst, this);
        }
        return new double[0];
    }

    /************************************************************
     *** HoeffdingTree implementation
     ************************************************************/
    protected LearningNode newLearningNode() {
        return newLearningNode(new double[0]);
    }

    protected LearningNode newLearningNode(double[] initialClassObservations) {
        return new LearningNodeNB(initialClassObservations);
    }

    public AttributeClassObserver newNominalClassObserver() {
        AttributeClassObserver nominalClassObserver = (AttributeClassObserver) getPreparedClassOption(this.nominalEstimatorOption);
        return (AttributeClassObserver) nominalClassObserver.copy();
    }

    public AttributeClassObserver newNumericClassObserver() {
        AttributeClassObserver numericClassObserver = (AttributeClassObserver) getPreparedClassOption(this.numericEstimatorOption);
        return (AttributeClassObserver) numericClassObserver.copy();
    }

    protected void attemptToSplit(ActiveLearningNode node, SplitNode parent, int parentBranch) {
        if (!node.observedClassDistributionIsPure()) {
            // get list of split suggestions and sort them
            SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
            AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, this);
            Arrays.sort(bestSplitSuggestions);

            // determine whether we should split based on Hoeffding bound or tie threshold
            boolean shouldSplit = false;
            if (bestSplitSuggestions.length > 2) {
                double hoeffdingBound = computeHoeffdingBound(
                        splitCriterion.getRangeOfMerit(node.getObservedClassDistribution()),
                        this.splitConfidenceOption.getValue(),
                        node.getWeightSeen()
                );
                AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                AttributeSplitSuggestion secondBestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 2];
                shouldSplit =
                        bestSuggestion.merit - secondBestSuggestion.merit > hoeffdingBound ||
                        hoeffdingBound < this.tieThresholdOption.getValue();
            }

            // split
            if (shouldSplit) {
                AttributeSplitSuggestion splitDecision = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                if (splitDecision.splitTest != null) {
                    SplitNode newSplit = new SplitNode(
                            splitDecision.splitTest,
                            node.getObservedClassDistribution(),
                            splitDecision.numSplits()
                    );

                    for (int i = 0; i < splitDecision.numSplits(); i++) {
                        Node newChild = newLearningNode(splitDecision.resultingClassDistributionFromSplit(i));
                        newSplit.setChild(i, newChild);
                    }

                    if (parent == null) {
                        this.treeRoot = newSplit;
                    } else {
                        parent.setChild(parentBranch, newSplit);
                    }
                }
            }
        }
    }

    public static double computeHoeffdingBound(double range, double confidence, double n) {
        return Math.sqrt(((range * range) * Math.log(1.0 / confidence))
                / (2.0 * n));
    }
}
