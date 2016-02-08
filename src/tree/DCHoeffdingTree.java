package tree;

import learningnode.ActiveLearningNode;
import learningnode.InactiveLearningNode;
import learningnode.LearningNode;
import learningnode.LearningNodeNB;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.DiscreteAttributeClassObserver;
import moa.classifiers.core.attributeclassobservers.NumericAttributeClassObserver;
import moa.classifiers.core.conditionaltests.InstanceConditionalTest;
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
    public IntOption gracePeriodOption = new IntOption(
            "gracePeriod",
            'g',
            "The number of instances a leaf should observe between split attempts.",
            200, 0, Integer.MAX_VALUE);

    public ClassOption numericEstimatorOption = new ClassOption("numericEstimator",
            'n', "Numeric estimator to use.", NumericAttributeClassObserver.class,
            "GaussianNumericAttributeClassObserver");

    public ClassOption nominalEstimatorOption = new ClassOption("nominalEstimator",
            'd', "Nominal estimator to use.", DiscreteAttributeClassObserver.class,
            "NominalAttributeClassObserver");

    public ClassOption splitCriterionOption = new ClassOption("splitCriterion",
            's', "Split criterion to use.", SplitCriterion.class,
            "InfoGainSplitCriterion");

    public FloatOption splitConfidenceOption = new FloatOption(
            "splitConfidence",
            'c',
            "The allowable error in split decision, values closer to 0 will take longer to decide.",
            0.0000001, 0.0, 1.0);

    public FloatOption tieThresholdOption = new FloatOption("tieThreshold",
            't', "Threshold below which a split will be forced to break ties.",
            0.05, 0.0, 1.0);

    // TODO pre-prune
    public FlagOption noPrePruneOption = new FlagOption("noPrePrune", 'p',
            "Disable pre-pruning.");

    // TODO binary split
    public FlagOption binarySplitsOption = new FlagOption("binarySplits", 'b',
            "Only allow binary splits.");

    public IntOption nbThresholdOption = new IntOption(
            "nbThreshold",
            'q',
            "The number of instances a leaf should observe before permitting Naive Bayes.",
            0, 0, Integer.MAX_VALUE);

    /************************************************************
     *** Variables
     ************************************************************/
    protected Node treeRoot;

    protected int decisionNodeCount;

    protected int activeLeafNodeCount;

    protected int inactiveLeafNodeCount;

    /************************************************************
     *** Methods from AbstractClassifier
     ************************************************************/
    @Override
    public void resetLearningImpl() {
        this.treeRoot = null;
        this.decisionNodeCount = 0;
        this.activeLeafNodeCount = 0;
        this.inactiveLeafNodeCount = 0;
        // TODO other options
//        this.inactiveLeafByteSizeEstimate = 0.0;
//        this.activeLeafByteSizeEstimate = 0.0;
//        this.byteSizeEstimateOverheadFraction = 1.0;
//        this.growthAllowed = true;
//        if (this.leafpredictionOption.getChosenIndex()>0) {
//            this.removePoorAttsOption = null;
//        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if (this.treeRoot == null) {
            this.treeRoot = newLearningNode();
            this.activeLeafNodeCount = 1;
        }
        FoundNode foundNode = this.treeRoot.filterInstanceToLeaf(inst, null, -1);
        Node leafNode = foundNode.node;
        if (leafNode == null) {
            leafNode = newLearningNode();
            foundNode.parent.setChild(foundNode.parentBranch, leafNode);
            this.activeLeafNodeCount++;
        }

        if (leafNode instanceof LearningNode) {
            LearningNode learningNode = (LearningNode) leafNode;
            learningNode.learnFromInstance(inst, this);
            if (learningNode instanceof ActiveLearningNode) {
                ActiveLearningNode activeLearningNode = (ActiveLearningNode) learningNode;
                double weightSeen = activeLearningNode.getWeightSeen();
                if (weightSeen - activeLearningNode.getWeightSeenAtLastSplitEvaluation()
                        >= this.gracePeriodOption.getValue()) {
                    attemptToSplit(activeLearningNode, foundNode.parent,
                            foundNode.parentBranch);
                    activeLearningNode.setWeightSeenAtLastSplitEvaluation(weightSeen);
                }
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
    public Node getTreeRoot() {
        return treeRoot;
    }

    protected LearningNode newLearningNode() {
        return newLearningNode(new double[0]);
    }

    // TODO other options for LearningNode
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

    protected void attemptToSplit(ActiveLearningNode node, SplitNode parent,
                                  int parentIndex) {
        if (!node.observedClassDistributionIsPure()) {
            SplitCriterion splitCriterion = (SplitCriterion) getPreparedClassOption(this.splitCriterionOption);
            AttributeSplitSuggestion[] bestSplitSuggestions = node.getBestSplitSuggestions(splitCriterion, this);
            Arrays.sort(bestSplitSuggestions);
            boolean shouldSplit = false;
            if (bestSplitSuggestions.length < 2) {
                shouldSplit = bestSplitSuggestions.length > 0;
            } else {
                double hoeffdingBound = computeHoeffdingBound(splitCriterion.getRangeOfMerit(node.getObservedClassDistribution()),
                        this.splitConfidenceOption.getValue(), node.getWeightSeen());
                AttributeSplitSuggestion bestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                AttributeSplitSuggestion secondBestSuggestion = bestSplitSuggestions[bestSplitSuggestions.length - 2];
                if ((bestSuggestion.merit - secondBestSuggestion.merit > hoeffdingBound)
                        || (hoeffdingBound < this.tieThresholdOption.getValue())) {
                    shouldSplit = true;
                }
                // }

                // TODO remove poor attributes
//                if ((this.removePoorAttsOption != null)
//                        && this.removePoorAttsOption.isSet()) {
//                    Set<Integer> poorAtts = new HashSet<Integer>();
//                    // scan 1 - add any poor to set
//                    for (int i = 0; i < bestSplitSuggestions.length; i++) {
//                        if (bestSplitSuggestions[i].splitTest != null) {
//                            int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
//                            if (splitAtts.length == 1) {
//                                if (bestSuggestion.merit
//                                        - bestSplitSuggestions[i].merit > hoeffdingBound) {
//                                    poorAtts.add(new Integer(splitAtts[0]));
//                                }
//                            }
//                        }
//                    }
//                    // scan 2 - remove good ones from set
//                    for (int i = 0; i < bestSplitSuggestions.length; i++) {
//                        if (bestSplitSuggestions[i].splitTest != null) {
//                            int[] splitAtts = bestSplitSuggestions[i].splitTest.getAttsTestDependsOn();
//                            if (splitAtts.length == 1) {
//                                if (bestSuggestion.merit
//                                        - bestSplitSuggestions[i].merit < hoeffdingBound) {
//                                    poorAtts.remove(new Integer(splitAtts[0]));
//                                }
//                            }
//                        }
//                    }
//                    for (int poorAtt : poorAtts) {
//                        node.disableAttribute(poorAtt);
//                    }
//                }
            }
            if (shouldSplit) {
                AttributeSplitSuggestion splitDecision = bestSplitSuggestions[bestSplitSuggestions.length - 1];
                if (splitDecision.splitTest == null) {
                    // preprune - null wins
                    deactivateLearningNode(node, parent, parentIndex);
                } else {
                    SplitNode newSplit = newSplitNode(splitDecision.splitTest,
                            node.getObservedClassDistribution(),splitDecision.numSplits() );
                    for (int i = 0; i < splitDecision.numSplits(); i++) {
                        Node newChild = newLearningNode(splitDecision.resultingClassDistributionFromSplit(i));
                        newSplit.setChild(i, newChild);
                    }
                    this.activeLeafNodeCount--;
                    this.decisionNodeCount++;
                    this.activeLeafNodeCount += splitDecision.numSplits();
                    if (parent == null) {
                        this.treeRoot = newSplit;
                    } else {
                        parent.setChild(parentIndex, newSplit);
                    }
                }
                // TODO manage memory
//                enforceTrackerLimit();
            }
        }
    }

    protected void deactivateLearningNode(ActiveLearningNode toDeactivate,
                                          SplitNode parent, int parentBranch) {
        Node newLeaf = new InactiveLearningNode(toDeactivate.getObservedClassDistribution());
        if (parent == null) {
            this.treeRoot = newLeaf;
        } else {
            parent.setChild(parentBranch, newLeaf);
        }
        this.activeLeafNodeCount--;
        this.inactiveLeafNodeCount++;
    }

    protected SplitNode newSplitNode(InstanceConditionalTest splitTest,
                                     double[] classObservations, int size) {
        return new SplitNode(splitTest, classObservations, size);
    }

    public static double computeHoeffdingBound(double range, double confidence, double n) {
        return Math.sqrt(((range * range) * Math.log(1.0 / confidence))
                / (2.0 * n));
    }
}
