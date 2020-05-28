package differentclassification;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.classifiers.lazy.IBk;
import weka.classifiers.functions.SMO;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.ZeroR;
 

public class DifferentClassification {

   	public static void main(String args[]) throws Exception{
		//load dataset
		DataSource source = new DataSource("F:\\THESIS\\BCI_data\\eye state\\emotivdata.arff");
		Instances dataset = source.getDataSet();	
		//set class index to the last attribute
		dataset.setClassIndex(dataset.numAttributes()-1);
		//create and build the classifier!
                
                IBk ibk=new IBk();
                ibk.buildClassifier(dataset);
                System.out.print(ibk.getCapabilities().toString());
                System.out.println("KNN classifier completed");
                System.out.println();
                
                OneR oner=new OneR();
                oner.buildClassifier(dataset);
                System.out.print(oner.getCapabilities().toString());
                System.out.println("oner classifier completed");
                
                ZeroR zeror=new ZeroR();
                zeror.buildClassifier(dataset);
                System.out.print(zeror.getCapabilities().toString());
                System.out.println("zeror classifier completed");
                
                
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(dataset);
		//print out capabilities
		System.out.print(nb.getCapabilities().toString());
                System.out.println("NaiveBayes Classifier completed");
                System.out.println();
		
               
		SMO svm = new SMO();
		svm.buildClassifier(dataset);
		System.out.print(svm.getCapabilities().toString());
                System.out.println("Support vector matchine classifier completed");
		 System.out.println();
                 
               
                 
		J48 tree = new J48(); 
		tree.buildClassifier(dataset);
		System.out.print(tree.getCapabilities().toString());
                System.out.println("Decition tree C 4.5 classifier completed");
		 System.out.println();
		
	}
}
