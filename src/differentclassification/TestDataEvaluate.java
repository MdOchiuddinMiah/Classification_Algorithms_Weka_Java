package differentclassification;
import weka.core.Instances;
import java.util.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.functions.SMO;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.bayes.NaiveBayes;
public class TestDataEvaluate {
    
    	public static void main(String args[]) throws Exception{
		//load datasets
		DataSource source = new DataSource("F:\\THESIS\\BCI_data\\eye state\\emotivdata.arff");
		Instances dataset = source.getDataSet();	
		//set class index to the last attribute
		dataset.setClassIndex(dataset.numAttributes()-1);
		//create and build the classifier!
		J48 tree = new J48();
		tree.buildClassifier(dataset);
		
		Evaluation eval = new Evaluation(dataset);
		//Random rand = new Random(1);
		//int folds = 10;
		
	 
		//test dataset for evaluation
		DataSource source1 = new DataSource("F:\\THESIS\\BCI_data\\eye state\\emotivtestdata.arff");
		Instances testDataset = source1.getDataSet();
		//set class index to the last attribute
		testDataset.setClassIndex(testDataset.numAttributes()-1);
		//now evaluate model
		eval.evaluateModel(tree, testDataset);
                
                
                System.out.println("Using decision tree j48 c4.5");
                
                
		//eval.crossValidateModel(tree, testDataset, folds, rand);
		System.out.println(eval.toSummaryString("Evaluation results:\n", false));
		
		System.out.println("Correct % = "+eval.pctCorrect());
		System.out.println("Incorrect % = "+eval.pctIncorrect());
		System.out.println("MAE = "+eval.meanAbsoluteError());
		System.out.println("RMSE = "+eval.rootMeanSquaredError());
		System.out.println("RAE = "+eval.relativeAbsoluteError());
		System.out.println("RRSE = "+eval.rootRelativeSquaredError());
		System.out.println("Precision = "+eval.precision(1));
		System.out.println("Recall = "+eval.recall(1));
		System.out.println("fMeasure = "+eval.fMeasure(1));
		System.out.println("Error Rate = "+eval.errorRate());
	    //the confusion matrix
		System.out.println(eval.toMatrixString("=== Overall Confusion Matrix ===\n"));
	        
                
                
                System.out.println();
                System.out.println("--------------------------------------------------");
                System.out.println("using KNN classifier");
                Evaluation eval1 = new Evaluation(dataset);
               IBk ibk = new IBk();
		ibk.buildClassifier(dataset);
                eval1.evaluateModel(ibk, testDataset);
                //eval1.crossValidateModel(ibk, testDataset, folds, rand);
		System.out.println(eval1.toSummaryString("Evaluation results:\n", false));
		
		System.out.println("Correct % = "+eval1.pctCorrect());
		System.out.println("Incorrect % = "+eval1.pctIncorrect());
		System.out.println("MAE = "+eval1.meanAbsoluteError());
		System.out.println("RMSE = "+eval1.rootMeanSquaredError());
		System.out.println("RAE = "+eval1.relativeAbsoluteError());
		System.out.println("RRSE = "+eval1.rootRelativeSquaredError());
		System.out.println("Precision = "+eval1.precision(1));
		System.out.println("Recall = "+eval1.recall(1));
		System.out.println("fMeasure = "+eval1.fMeasure(1));
		System.out.println("Error Rate = "+eval1.errorRate());
	    //the confusion matrix
		System.out.println(eval1.toMatrixString("=== Overall Confusion Matrix ===\n"));
	                        
                
                
                    System.out.println();
                System.out.println("--------------------------------------------------");
                System.out.println("using Naive Bayes classifier");
                Evaluation eval2 = new Evaluation(dataset);
               NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(dataset);
                
                eval2.evaluateModel(nb, testDataset);
                //eval2.crossValidateModel(nb, testDataset, folds, rand);
		System.out.println(eval2.toSummaryString("Evaluation results:\n", false));
		
		System.out.println("Correct % = "+eval2.pctCorrect());
		System.out.println("Incorrect % = "+eval2.pctIncorrect());
		System.out.println("MAE = "+eval2.meanAbsoluteError());
		System.out.println("RMSE = "+eval2.rootMeanSquaredError());
		System.out.println("RAE = "+eval2.relativeAbsoluteError());
		System.out.println("RRSE = "+eval2.rootRelativeSquaredError());
		System.out.println("Precision = "+eval2.precision(1));
		System.out.println("Recall = "+eval2.recall(1));
		System.out.println("fMeasure = "+eval2.fMeasure(1));
		System.out.println("Error Rate = "+eval2.errorRate());
	    //the confusion matrix
		System.out.println(eval2.toMatrixString("=== Overall Confusion Matrix ===\n"));
	        
                
                
                
                    System.out.println();
                System.out.println("--------------------------------------------------");
                System.out.println("using OneR classifier");
                Evaluation eval3 = new Evaluation(dataset);
               OneR oner = new OneR();
		oner.buildClassifier(dataset);
                eval3.evaluateModel(oner, testDataset);
                //eval3.crossValidateModel(oner, testDataset, folds, rand);
		System.out.println(eval3.toSummaryString("Evaluation results:\n", false));
		
		System.out.println("Correct % = "+eval3.pctCorrect());
		System.out.println("Incorrect % = "+eval3.pctIncorrect());
		System.out.println("MAE = "+eval3.meanAbsoluteError());
		System.out.println("RMSE = "+eval3.rootMeanSquaredError());
		System.out.println("RAE = "+eval3.relativeAbsoluteError());
		System.out.println("RRSE = "+eval3.rootRelativeSquaredError());
		System.out.println("Precision = "+eval3.precision(1));
		System.out.println("Recall = "+eval3.recall(1));
		System.out.println("fMeasure = "+eval3.fMeasure(1));
		System.out.println("Error Rate = "+eval3.errorRate());
	    //the confusion matrix
		System.out.println(eval3.toMatrixString("=== Overall Confusion Matrix ===\n"));
	        
                
                
                
                
                    System.out.println();
                System.out.println("--------------------------------------------------");
                System.out.println("using support vector matchine classifier");
                Evaluation eval4 = new Evaluation(dataset);
               SMO smo = new SMO();
		smo.buildClassifier(dataset);
                
                eval4.evaluateModel(smo, testDataset);
                //eval4.crossValidateModel(smo, testDataset, folds, rand);
		System.out.println(eval4.toSummaryString("Evaluation results:\n", false));
		
		System.out.println("Correct % = "+eval4.pctCorrect());
		System.out.println("Incorrect % = "+eval4.pctIncorrect());
		System.out.println("MAE = "+eval4.meanAbsoluteError());
		System.out.println("RMSE = "+eval4.rootMeanSquaredError());
		System.out.println("RAE = "+eval4.relativeAbsoluteError());
		System.out.println("RRSE = "+eval4.rootRelativeSquaredError());
		System.out.println("Precision = "+eval4.precision(1));
		System.out.println("Recall = "+eval4.recall(1));
		System.out.println("fMeasure = "+eval4.fMeasure(1));
		System.out.println("Error Rate = "+eval4.errorRate());
	    //the confusion matrix
		System.out.println(eval4.toMatrixString("=== Overall Confusion Matrix ===\n"));
	        
                
                
                     System.out.println();
                System.out.println("--------------------------------------------------");
                System.out.println("using ZeroR classifier");
                Evaluation eval5 = new Evaluation(dataset);
               ZeroR zeror = new ZeroR();
		zeror.buildClassifier(dataset);
                
                eval5.evaluateModel(zeror, testDataset);
                //eval5.crossValidateModel(zeror, testDataset, folds, rand);
		System.out.println(eval5.toSummaryString("Evaluation results:\n", false));
		
		System.out.println("Correct % = "+eval5.pctCorrect());
		System.out.println("Incorrect % = "+eval5.pctIncorrect());
		System.out.println("MAE = "+eval5.meanAbsoluteError());
		System.out.println("RMSE = "+eval5.rootMeanSquaredError());
		System.out.println("RAE = "+eval5.relativeAbsoluteError());
		System.out.println("RRSE = "+eval5.rootRelativeSquaredError());
		System.out.println("Precision = "+eval5.precision(1));
		System.out.println("Recall = "+eval5.recall(1));
		System.out.println("fMeasure = "+eval5.fMeasure(1));
		System.out.println("Error Rate = "+eval5.errorRate());
	    //the confusion matrix
		System.out.println(eval5.toMatrixString("=== Overall Confusion Matrix ===\n"));
	        
                
                
                
                
	}
    
    
}
