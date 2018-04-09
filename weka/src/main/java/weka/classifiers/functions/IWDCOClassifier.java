/**
 * 
 */
package weka.classifiers.functions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Vector;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import weka.classifiers.RandomizableClassifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @author Shubham Kumaram
 *
 */
public class IWDCOClassifier extends RandomizableClassifier {

	private static final long serialVersionUID = 7043829761230254668L;
	private Instances instances;
	
	//Number of Inputs to the Neural Network
	private int numAttributes;
	
	//Number of output classes
	private int numClasses;
	
	//Number of nodes in the first and only hidden layer
	private int numNodesInHiddenLayer;
	
	private int precision;
	
	//Total number of trainable weights in the system
	private int numWeights;
	
	//Number of epochs to run the ENTIRE DATASET on
	private int numEpochs;
	
	//Number of iterations that IWDs make on ONE TRAINING EXAMPLE of the DATASET
	private int numIterations;

	/** IWD Parameters */
	private double minSoil;
	private double maxSoil;
	
	/** Random number generator */
	private Random random;
	private int randomSeed;
	private IWD bestIWD;
	private Vector <IWD> IWDs;
	
	/**
	 * soilValues maps a pair (j,i), where i is the index of current
	 * weight w_i, and j is the index of selected value of w_i, to all the
	 * paths which emanate from node (j,i) in the IWD graph
	 */
	private ArrayList <Double> edgeSoils;
	private int currentIterationBestPathIndices[];
	private int bestPathIndices[];
	private double globalLeastError;
	private int minWeight;
	private int maxWeight;
	
	/**
	 * Constructor
	 */
	
	public IWDCOClassifier() {
		instances = null;
		numAttributes = 0;
		numClasses = 0;
		numNodesInHiddenLayer = 10;
		numWeights = 0;
		precision = 32;
		numEpochs = 1;
		numIterations = 1;
		
		minSoil = 2000;
		maxSoil = 10000;
		minWeight = -10;
		maxWeight = 10;

		random = null;
		randomSeed = 0;
		
		bestIWD = null;
		IWDs = new Vector<IWD>();
		edgeSoils = new ArrayList<Double>();
		
		currentIterationBestPathIndices = null;
		bestPathIndices = null;
		globalLeastError = Double.POSITIVE_INFINITY;
	}
	/* (non-Javadoc)
	 * @see weka.classifiers.Classifier#buildClassifier(weka.core.Instances)
	 */
	@Override
	public void buildClassifier(Instances data) throws Exception {
		initializeClassifier(data);
		int numInstance = 0;
		while(numEpochs>0) {
			for (Instance instance : instances) {
				System.out.print("Instance "+numInstance++);
				for (int i=0; i<numIterations; i++) {
					System.out.println("\tIteration "+i);
					double leastError = makeIWDJourney(instance);
					if (leastError < globalLeastError) {
						globalLeastError = leastError;
						bestPathIndices = currentIterationBestPathIndices;
					}
					updateGlobalSoil();
				}
			}
			numEpochs--;
		}
	}

	private void initializeClassifier(Instances data) throws Exception {
		getCapabilities().testWithFail(data);
		
		data = new Instances(data);
		data.deleteWithMissingClass();
		
		instances = new Instances(data);
		random = new Random(randomSeed);
		instances.randomize(random);
		
		numAttributes = instances.numAttributes() - 1;
		numClasses = instances.numClasses();
		numWeights = numNodesInHiddenLayer * (numAttributes + numClasses - 1);
		
		initializePathSoil();
		placeIWDs();
	}

	private void initializePathSoil() {
		double initSoil = 5000;
		for (int i=0; i<precision*numWeights*2; i++) {
			edgeSoils.add(initSoil);
		}

	}

	private void placeIWDs() {
		int numIWDs = 1000;
		if (IWDs.isEmpty()) {
			for (int i=0; i<numIWDs; i++) {
				IWD iwd = new IWD(0);
				IWDs.add(iwd);
			}
		} else {
			for (int i=0;i<numIWDs; i++) {
				IWDs.get(i).currentPosition=0;
			}
		}

	}
	
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		double result[] = new double[numClasses];
		result[1] = getPrediction(instance, bestPathIndices);
		result[0] = 1-result[1];
		System.out.println(result[0]+"\t"+result[1]);
		return result;
	}
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		runClassifier(new IWDClassifier(), args);
	}

	private double getPrediction(Instance instance, int[] iWDPath) {
		ArrayList <Double> weightValues = setWeightValues(iWDPath);
		
		DenseMatrix firstWeightMatrix = new DenseMatrix(numAttributes , numNodesInHiddenLayer);
		double instanceArray[] = new double[numAttributes];
		System.arraycopy(instance.toDoubleArray(), 0, instanceArray, 0, numAttributes);
		DenseVector inputVector = new DenseVector(instanceArray, false);
		
		int i = 0 ;
		for ( int r = 0 ; r < numAttributes ; r++ ) {
			for ( int c = 0 ; c < numNodesInHiddenLayer ; c++ ) {
				firstWeightMatrix.set(r, c, weightValues.get(i));
				i++;
			}
		}
		
		DenseVector hiddenLayerOutput = new DenseVector(numNodesInHiddenLayer) ;
		firstWeightMatrix.transMult(inputVector, hiddenLayerOutput);
		
		double output = 0 ; 
		int ctr = 0 ;
		for ( ; i < numWeights ; i++ ) {
			double activatedHiddenLayerOutput = Math.tanh(hiddenLayerOutput.get(ctr));
			output += activatedHiddenLayerOutput * weightValues.get(i);
			ctr++ ; 
		}
		
		output = 1d/(1+Math.pow(Math.E, -output));
		return output ; 
	}
	
	private ArrayList<Double> setWeightValues(int IWDPath[]) {
		ArrayList <Double> weightValues = new ArrayList<Double>(numWeights);
		double actualRange = maxWeight - minWeight;
		double rawRange = Math.pow(2, precision);
		for (int i=0; i<numWeights; i++) {
			//convert to decimal
			int rawDecimal = 0;
			for (int bitIndex = 0; bitIndex < precision; bitIndex++) {
				rawDecimal = rawDecimal*2 + IWDPath[bitIndex];
			}
			//scale
			double scaledWeight = (double) rawDecimal * actualRange / rawRange;
			//offset
			scaledWeight += minWeight;
			weightValues.add(scaledWeight);
		}
		return weightValues;
	}
	
	private double makeIWDJourney(Instance instance) {
		double leastError = Double.POSITIVE_INFINITY;
		int numIWDs = IWDs.size();
		for (int i=0 ; i<numIWDs; i++) {
			IWD iwd = IWDs.get(i);
			int IWDPath[] = iwd.traversePaths();
			double errorAfterMutations = mutations(instance, IWDPath);
			
			/*
			 * Since quality is inverse of error, selecting path
			 * with least error is equivalent to selecting path with
			 * greatest quality
			 */
			if (errorAfterMutations < leastError) {
				leastError  = errorAfterMutations;
				currentIterationBestPathIndices = IWDPath;
				bestIWD = iwd;
			}
		}
		return leastError;
	}

	private double mutations(Instance instance, int[] IWDPath) {
		double prediction = getPrediction(instance, IWDPath);
		double currentError = calculateError(instance, prediction);
		for (int i=0;i<100;i++) {
			int indexOfEdgeToBeMutated = random.nextInt(numWeights);
			IWDPath[indexOfEdgeToBeMutated] = (IWDPath[indexOfEdgeToBeMutated]==0) ? 1 : 0;
			prediction = getPrediction(instance, IWDPath);
			double error = calculateError(instance, prediction);
			if (error < currentError) {
				currentError = error;
			} else {
				IWDPath[indexOfEdgeToBeMutated] = (IWDPath[indexOfEdgeToBeMutated]==0) ? 1 : 0;
			}
		}
		return prediction; 
	}
	private void updateGlobalSoil() {
		for (int i=0;i<numWeights-1;i++) {
			double currentSoil = edgeSoils.get(2*i+currentIterationBestPathIndices[i]);
			double tempSoil = 1.1*currentSoil - (0.01 * bestIWD.soil) / (precision * numWeights);
			double newSoil = Math.min(Math.max(tempSoil, minSoil), maxSoil);
			edgeSoils.set(2*i+currentIterationBestPathIndices[i], newSoil);
		}
	}

	private double calculateError(Instance instance, double prediction) {
		double actualClass = instance.classValue();
		double error = actualClass - prediction;
		return Math.abs(error);
	}
	
	/**
	 * @return the numNodesInHiddenLayer
	 */
	public int getNumNodesInHiddenLayer() {
		return numNodesInHiddenLayer;
	}
	/**
	 * @param numNodesInHiddenLayer the numNodesInHiddenLayer to set
	 */
	public void setNumNodesInHiddenLayer(int numNodesInHiddenLayer) {
		this.numNodesInHiddenLayer = numNodesInHiddenLayer;
	}
	
	public String numNodesInHiddenLayerTipText() {
		return "Number of nodes in the hidden layer";
	}

	/**
	 * @return the precision
	 */
	public int getPrecision() {
		return precision;
	}
	/**
	 * @param precision the precision to set
	 */
	public void setPrecision(int precision) {
		this.precision = precision;
	}
	
	public String precisionTipText() {
		return "Set number of values of weights to consider";
	}

	/**
	 * @return the numEpochs
	 */
	public int getNumEpochs() {
		return numEpochs;
	}
	/**
	 * @param numEpochs the numEpochs to set
	 */
	public void setNumEpochs(int numEpochs) {
		this.numEpochs = numEpochs;
	}
	
	public String numEpochsTipText() {
		return "Set number of epochs";
	}

	/**
	 * @return the numIterations
	 */
	public int getNumIterations() {
		return numIterations;
	}
	/**
	 * @param numIterations the numIterations to set
	 */
	public void setNumIterations(int numIterations) {
		this.numIterations = numIterations;
	}
	
	public String numIterationsTipText() {
		return "Set number of iterations for each instance";
	}

	/**
	 * @return the minWeight
	 */
	public int getMinWeight() {
		return minWeight;
	}
	/**
	 * @param minWeight the minWeight to set
	 */
	public void setMinWeight(int minWeight) {
		this.minWeight = minWeight;
	}
	
	public String minWeightTipText() {
		return "Lower limit of weight range";
	}

	/**
	 * @return the maxWeight
	 */
	public int getMaxWeight() {
		return maxWeight;
	}
	/**
	 * @param maxWeight the maxWeight to set
	 */
	public void setMaxWeight(int maxWeight) {
		this.maxWeight = maxWeight;
	}

	public String maxWeightTipText() {
		return "Upper limit of weight range";
	}

	class IWD implements Serializable{
		
		private static final long serialVersionUID = -8492864630943739397L;
		private double soil;
		private int currentPosition;
		double delSoil = 0.001;
		
		private int selectedPathIndices[];
		
		IWD(int position) {
			currentPosition = position;
			soil = 10000;
			
			selectedPathIndices = new int[precision*numWeights];
		}
		
		public int[] traversePaths() {
			while (currentPosition < numWeights) {
				double soilValues[] = new double[2];
				soilValues[0] = edgeSoils.get(2*currentPosition);
				soilValues[1] = edgeSoils.get(2*currentPosition+1);

				int index = getNextIWDJump(soilValues);
				double soilOfSelectedPath = soilValues[index];
				edgeSoils.set(2*currentPosition+index, 1.1*soilOfSelectedPath - 0.01*delSoil);
				soil += delSoil;

				selectedPathIndices[currentPosition] = index;
				currentPosition = currentPosition+1;
			}
			return selectedPathIndices;
		}
		
		// Return g_soil(i,j) while standing at node i and comparing with nodes of next layer. 
		private double[] getGSoil(double[] soil_ij) {
			double new_soil_ij[] = new double[2];
			//Calculating minimum ij
			double minSoil = Double.POSITIVE_INFINITY;
			for(double ii : soil_ij) {
					if( ii < minSoil ) {
						minSoil = ii;
					}
			}
			if ( minSoil < 0 ) {
				//All soil values become (soil-minSoil)
				new_soil_ij[0] = soil_ij[0] - minSoil;
				new_soil_ij[1] = soil_ij[1] - minSoil;
			}
			else {
				//All soil values become (soil-minSoil)
				new_soil_ij[0] = soil_ij[0];
				new_soil_ij[1] = soil_ij[1];
			}
			return new_soil_ij;
		}


		// Return f_soil(i,j) while standing at node i and comparing with nodes of next layer. 
		private double[] getFSoil(double[] g_soil_ij) {
			double epsilon_s = 0.0001;
			//Calculating f_soil_ij from g_soil_ij
			g_soil_ij[0] = 1 / ( epsilon_s + g_soil_ij[0]); 
			g_soil_ij[1] = 1 / ( epsilon_s + g_soil_ij[1]); 
			return g_soil_ij;
		}
		
		public int getNextIWDJump(double soilValues[]) {
			double sigma_f_soil = 0;
			double zero_probability = 0;
			double[] f_soil_ij = getFSoil(getGSoil(soilValues));
			double random_number = random.nextDouble();

			sigma_f_soil = f_soil_ij[0] + f_soil_ij[1];
			zero_probability = f_soil_ij[0] / sigma_f_soil;
			
			if (random_number < zero_probability) return 0;
			else return 1;
		}
	}

}