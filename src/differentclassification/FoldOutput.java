package differentclassification;
import weka.core.Instances;
import java.util.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.lazy.IBk;
import weka.classifiers.functions.SMO;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;

public class FoldOutput {
  
    public static void main(String args[]) throws Exception{
		//load dataset
		DataSource source = new DataSource("F:\\THESIS\\BCI_data\\eye state\\emotivdata.arff");
		Instances dataset = source.getDataSet();	
		//set class index to the last attribute
		dataset.setClassIndex(dataset.numAttributes()-1);

		

		int seed = 1;
		int folds = 10;
		// randomize data
		Random rand = new Random(seed);
		//create random dataset
		Instances randData = new Instances(dataset);
		randData.randomize(rand);
		//stratify	    
		if (randData.classAttribute().isNominal())
			randData.stratify(folds);

                
                       
                ZeroR zeror=new ZeroR();
                  System.out.println("10 fold using ZeroR classifier");
		for (int n = 0; n < folds; n++) {
			Evaluation eval = new Evaluation(randData);
			//get the folds	      
			Instances train = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);	      
			// build and evaluate classifier	     
			zeror.buildClassifier(train);
			eval.evaluateModel(zeror, test);

			// output evaluation
			System.out.println();
			System.out.println("Correct % = "+eval.pctCorrect());
			System.out.println("Incorrect % = "+eval.pctIncorrect());
                       
			 
		}
                
                
                
                
                
                //create the classifier
		NaiveBayes nb = new NaiveBayes();
		// perform cross-validation
                System.out.println("10 fold using naive bayes");
		for (int n = 0; n < folds; n++) {
			Evaluation eval = new Evaluation(randData);
			//get the folds	      
			Instances train = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);	      
			// build and evaluate classifier	     
			nb.buildClassifier(train);
			eval.evaluateModel(nb, test);

			// output evaluation
			System.out.println();
			System.out.println("Correct % = "+eval.pctCorrect());
			System.out.println("Incorrect % = "+eval.pctIncorrect());
                       
			 
		}
    
                
                System.out.println("--------------------------------------------------");
                IBk ibk=new IBk();
                  System.out.println("10 fold KNN");
		for (int n = 0; n < folds; n++) {
			Evaluation eval = new Evaluation(randData);
			//get the folds	      
			Instances train = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);	      
			// build and evaluate classifier	     
			ibk.buildClassifier(train);
			eval.evaluateModel(ibk, test);

			// output evaluation
			System.out.println();
			System.out.println("Correct % = "+eval.pctCorrect());
			System.out.println("Incorrect % = "+eval.pctIncorrect());
                       
			 
		}
                
                
                
                  System.out.println("--------------------------------------------------");
                SMO smo=new SMO();
                  System.out.println("10 fold support vector matchine");
		for (int n = 0; n < folds; n++) {
			Evaluation eval = new Evaluation(randData);
			//get the folds	      
			Instances train = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);	      
			// build and evaluate classifier	     
			smo.buildClassifier(train);
			eval.evaluateModel(smo, test);

			// output evaluation
			System.out.println();
			System.out.println("Correct % = "+eval.pctCorrect());
			System.out.println("Incorrect % = "+eval.pctIncorrect());
                       
			 
		}
                
                
                
                   System.out.println("--------------------------------------------------");
                J48 j48=new J48();
                  System.out.println("10 fold decision tree c 4.5");
		for (int n = 0; n < folds; n++) {
			Evaluation eval = new Evaluation(randData);
			//get the folds	      
			Instances train = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);	      
			// build and evaluate classifier	     
			j48.buildClassifier(train);
			eval.evaluateModel(j48, test);

			// output evaluation
			System.out.println();
			System.out.println("Correct % = "+eval.pctCorrect());
			System.out.println("Incorrect % = "+eval.pctIncorrect());
                       
			 
		}
                
                
                       System.out.println("--------------------------------------------------");
                OneR oner=new OneR();
                  System.out.println("10 fold oneR");
		for (int n = 0; n < folds; n++) {
			Evaluation eval = new Evaluation(randData);
			//get the folds	      
			Instances train = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);	      
			// build and evaluate classifier	     
			oner.buildClassifier(train);
			eval.evaluateModel(oner, test);

			// output evaluation
			System.out.println();
			System.out.println("Correct % = "+eval.pctCorrect());
			System.out.println("Incorrect % = "+eval.pctIncorrect());
                       
			 
		}
                
                
    }
    
}
