# dc-hoeffing-tree
An implementation of Hoeffding Tree

**Example** (need to download the dataset [covtypeNorm.arff](http://archive.ics.uci.edu/ml/machine-learning-databases/covtype/) first)
<br/>
javac -cp moa.jar $(find * -name \*.java)
<br/>
java -classpath moa.jar:src/ -javaagent:sizeofag.jar moa.DoTask \
<br/>
"EvaluatePrequential -l tree.DCHoeffdingTree -s (ArffFileStream -f covtypeNorm.arff) -i 581000"
