/**
 * 
 */
package weka.classifiers.functions;

import java.util.Map;
import java.util.Random;
import java.util.Vector;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import weka.classifiers.RandomizableClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

/**
 * @author smriti srivastava
 *
 */
public class IWDClassifier extends RandomizableClassifier {

	private static final long serialVersionUID = 7043829761230254668L;
	private Instances instances;
	private int numAttributes;
	private int numClasses;
	private int numNodesInHiddenLayer;
	private int precision;
	private int numWeights;

	private double HUD;
	private int a_v;
	private double b_v;
	private int c_v;
	private int a_s;
	private double b_s;
	private int c_s;
	private double rho_n;
	private double rho_IWD;
	
	/** Random number generator */
	private Random random;
	private int randomSeed;
	private IWD bestIWD;
	private Vector <IWD> IWDs;
	private Vector <Vector <Double >> weightValues;
	
	/**
	 * soilValues maps a pair (j,i), where i is the index of current
	 * weight w_i, and j is the index of selected value of w_i, to all the
	 * paths which emanate from node (j,i) in the IWD graph
	 */
	private Map < Pair, Vector<Double>> soilValues;
	private int currentIterationBestPathIndices[];
	private int bestPathIndices[];
	private double globalLeastError;
	
	/**
	 * Constructor
	 */
	
	public IWDClassifier() {
		instances = null;
		numAttributes = 0;
		numClasses = 0;
		numNodesInHiddenLayer = 0;
		numWeights = 0;
		precision = 0;
		
		HUD = 0;
		a_v = 1;
		b_v = 0.01;
		c_v = 1;
		a_s = 1;
		b_s = 0.01;
		c_s = 1;
		rho_n = 0.9;
		rho_IWD = 0.9;
		
		random = null;
		randomSeed = 0;
		
		bestIWD = null;
		IWDs = new Vector<IWD>();
		weightValues = new Vector <Vector < Double> >();
		soilValues = new HashMap <Pair, Vector<Double>>();
		
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
		for (Instance instance : instances) {
			double leastError = makeIWDJourney(instance);
			if (leastError < globalLeastError) {
				globalLeastError = leastError;
				bestPathIndices = currentIterationBestPathIndices;
			}
			updateGlobalSoil();
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
		numWeights = numNodesInHiddenLayer * (numAttributes + numClasses);
		
		initializeWeights();
		placeIWDs();
	}

	private void initializeWeights() {
		// TODO Auto-generated method stub
	}

	private void placeIWDs() {
		//One IWD is placed at all probable values of the first weight
		int numIwds = precision;
		if (IWDs.isEmpty()) {
			for (int i=0; i<numIwds; i++) {
				IWD iwd = new IWD(new Pair(i, 0));
				IWDs.add(iwd);
			}
		} else {
			for (int i=0;i<numIwds; i++) {
				IWDs.get(i).currentPosition.x=i;
				IWDs.get(i).currentPosition.y=0;
			}
		}
	}
	
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return super.distributionForInstance(instance);
	}
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		runClassifier(new IWDClassifier(), args);
	}

	private double getPrediction(Instance instance, int[] iWDPath) {
		// TODO Auto-generated method stub
		return 0;
	}

	private double makeIWDJourney(Instance instance) {
		double leastError = Double.POSITIVE_INFINITY;
		for (IWD i : IWDs) {
			int IWDPath[] = i.traversePaths();
			double prediction = getPrediction(instance, IWDPath);
			double currentIWDPathError = calculateError(instance, prediction);
			
			/*
			 * Since quality is inverse of error, selecting path
			 * with least error is equivalent to selecting path with
			 * greatest quality
			 */
			if (currentIWDPathError < leastError) {
				leastError  = currentIWDPathError;
				currentIterationBestPathIndices = IWDPath;
				bestIWD = i;
			}
		}
		return leastError;
	}
	
	private void updateGlobalSoil() {
		for (int i=0;i<weightValues.size()-1;i++) {
			Pair nodePosition = new Pair(i, currentIterationBestPathIndices[i]);
			double currentSoil = soilValues.get(nodePosition).
					get(currentIterationBestPathIndices[i+1]);
			double newSoil = (1 + rho_IWD) * currentSoil
					- rho_IWD * (1/(numWeights - 1))*bestIWD.soil;
			soilValues.get(nodePosition).insertElementAt(newSoil, bestPathIndices[i+1]);

		}
	}

	private double calculateError(Instance instance, double prediction) {
		double actualClass = instance.classValue();
		double error = actualClass - prediction;
		return error*error;
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

	class IWD {
		
		private double soil;
		private double velocity;
		private Pair currentPosition;
		
		private int selectedWeightIndices[];
		
		IWD(Pair position) {
			currentPosition = position;
			velocity = 200;
			soil = 10000;
			
			selectedWeightIndices = new int[numWeights];
			selectedWeightIndices[0] = currentPosition.x;
		}
		
		public int[] traversePaths() {
			while (currentPosition.y != weightValues.size()) {
				Vector <Double> soilValuesOfPaths = soilValues.get(currentPosition);
				int index = getNextIWDJump(soilValuesOfPaths);

				double soilOfSelectedPath = soilValuesOfPaths.get(index);
				velocity += a_v/(b_v + c_v * soilOfSelectedPath*soilOfSelectedPath);

				double time = HUD/velocity;
				double delSoil = a_s/(b_s + c_s * time*time);
				soilOfSelectedPath = (1 - rho_n)*soilOfSelectedPath
						- rho_n * delSoil;
				soil += delSoil;

				currentPosition.x = index;
				currentPosition.y++;
				selectedWeightIndices[currentPosition.y] = currentPosition.x;
			}
			return selectedWeightIndices;
		}
		
		// Return g_soil(i,j) while standing at node i and comparing with nodes of next layer. 
		private Vector<Double> getGSoil(Vector<Double> soil_ij) {
			soil_ij = new Vector<Double>(soil_ij);
			//Calculating minimum ij
			double minSoil = Double.POSITIVE_INFINITY;
			for(Double ii : soil_ij) {
					if( Double.compare( ii , minSoil ) < 0 ) {
						minSoil = ii ;
					}
			}
			if ( minSoil < 0 ) {
				//All soil values become (soil-minSoil)
				for (int i=0;i<soil_ij.size();i++) {
					Double ii = soil_ij.get(i);
					ii = ii - minSoil;
					soil_ij.set(i, ii);
				}
			}
			return soil_ij;
		}


		// Return f_soil(i,j) while standing at node i and comparing with nodes of next layer. 
		private Vector<Double> getFSoil(Vector<Double> g_soil_ij) {
			double epsilon_s = 0.000001;
			// Lesson learnt : for each makes a copy
			for(int i=0;i<g_soil_ij.size();i++) {
				//Calculating f_soil_ij from g_soil_ij
				Double ii = g_soil_ij.get(i);
				ii = 1 / ( epsilon_s + ii); 
				g_soil_ij.set(i, ii);
			}
			return g_soil_ij;
		}
		
		//Return best next jump. Should calculate P(soil(i,j)) but not needed.
		public int getNextIWDJump(Vector<Double> soil_ij) {
			Vector<Double> f_soil_ij = getFSoil(getGSoil(soil_ij));
			int index = 0;
			int indexMax = 0;
			double maxF = Double.NEGATIVE_INFINITY;
			for(Double ii : f_soil_ij) {
				if( ii > maxF) {
					maxF = ii;
					indexMax = index; 
				}
				index++;
			}
			return indexMax;
		}
	}


	class Pair {
		int x;
		int y;
		
		Pair(int x, int y) { this.x = x; this.y = y;}

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + getOuterType().hashCode();
			result = prime * result + x;
			result = prime * result + y;
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj)
				return true;
			if (obj == null)
				return false;
			if (getClass() != obj.getClass())
				return false;
			Pair other = (Pair) obj;
			if (!getOuterType().equals(other.getOuterType()))
				return false;
			if (x != other.x)
				return false;
			if (y != other.y)
				return false;
			return true;
		}

		private IWDClassifier getOuterType() {
			return IWDClassifier.this;
		}
		
	}
	
}
