package differentclassification;


import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.functions.SMO;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;


public class ClassifyUnknownInstances {
    public static void main(String args[]) throws Exception{
		//load training dataset
		DataSource source = new DataSource("F:\\THESIS\\BCI_data\\eye state\\emotivdata.arff");
		Instances trainDataset = source.getDataSet();	
		//set class index to the last attribute
		trainDataset.setClassIndex(trainDataset.numAttributes()-1);

		//build model
		J48 j48 = new J48();
		j48.buildClassifier(trainDataset);
		//output model
		 

		//load new dataset
		DataSource source1 = new DataSource("F:\\THESIS\\BCI_data\\eye state\\emotivtestdata1.arff");
		Instances testDataset = source1.getDataSet();	
		//set class index to the last attribute
		testDataset.setClassIndex(testDataset.numAttributes()-1);

		//loop through the new dataset and make predictions
                System.out.println("first three are 0 and last three are 1 from training data");
		System.out.println("===================");
		System.out.println("Actual Class, j48 Predicted");
		for (int i = 0; i < testDataset.numInstances(); i++) {
			//get class double value for current instance
			double actualValue = testDataset.instance(i).classValue();

			//get Instance object of current instance
			Instance newInst = testDataset.instance(i);
			//call classifyInstance, which returns a double value for the class
			double predSMO = j48.classifyInstance(newInst);
                         
                        
                        
			System.out.println("    "+(int)actualValue+"              "+(int)predSMO);
		}
                
                IBk ibk=new IBk();
                ibk.buildClassifier(trainDataset);
                
                System.out.println();
                System.out.println("===================");
		System.out.println("Actual Class, KNN Predicted");
		for (int i = 0; i < testDataset.numInstances(); i++) {
			//get class double value for current instance
			double actualValue = testDataset.instance(i).classValue();

			//get Instance object of current instance
			Instance newInst = testDataset.instance(i);
			//call classifyInstance, which returns a double value for the class
			double predSMO = ibk.classifyInstance(newInst);
                         
                        
                        
			System.out.println("    "+(int)actualValue+"              "+(int)predSMO);
		}
                
                
                
                
                   NaiveBayes nb=new NaiveBayes();
                nb.buildClassifier(trainDataset);
                
                System.out.println();
                System.out.println("===================");
		System.out.println("Actual Class, naivebayes Predicted");
		for (int i = 0; i < testDataset.numInstances(); i++) {
			//get class double value for current instance
			double actualValue = testDataset.instance(i).classValue();

			//get Instance object of current instance
			Instance newInst = testDataset.instance(i);
			//call classifyInstance, which returns a double value for the class
			double predSMO = nb.classifyInstance(newInst);
                         
                        
                        
			System.out.println("    "+(int)actualValue+"                "+(int)predSMO);
		}
                
                
                
                
                   SMO smo=new SMO();
                smo.buildClassifier(trainDataset);
                    System.out.println();
                System.out.println("===================");
		System.out.println("Actual Class, smo Predicted");
		for (int i = 0; i < testDataset.numInstances(); i++) {
			//get class double value for current instance
			double actualValue = testDataset.instance(i).classValue();

			//get Instance object of current instance
			Instance newInst = testDataset.instance(i);
			//call classifyInstance, which returns a double value for the class
			double predSMO = smo.classifyInstance(newInst);
                         
                        
                        
			System.out.println("    "+(int)actualValue+"                "+(int)predSMO);
		}
                
                
                
                   
                  ZeroR zeror=new ZeroR();
                zeror.buildClassifier(trainDataset);
                    System.out.println();
                System.out.println("===================");
		System.out.println("Actual Class, Zeror Predicted");
		for (int i = 0; i < testDataset.numInstances(); i++) {
			//get class double value for current instance
			double actualValue = testDataset.instance(i).classValue();

			//get Instance object of current instance
			Instance newInst = testDataset.instance(i);
			//call classifyInstance, which returns a double value for the class
			double predSMO = zeror.classifyInstance(newInst);
                         
                        
                        
			System.out.println("    "+(int)actualValue+"                "+(int)predSMO);
		}
                
                
                
                   
                   OneR oner=new OneR();
                oner.buildClassifier(trainDataset);
                    System.out.println();
                System.out.println("===================");
		System.out.println("Actual Class, oner Predicted");
		for (int i = 0; i < testDataset.numInstances(); i++) {
			//get class double value for current instance
			double actualValue = testDataset.instance(i).classValue();

			//get Instance object of current instance
			Instance newInst = testDataset.instance(i);
			//call classifyInstance, which returns a double value for the class
			double predSMO = oner.classifyInstance(newInst);
                         
                        
                        
			System.out.println("    "+(int)actualValue+"                "+(int)predSMO);
		}
                
                
    }
    
}
