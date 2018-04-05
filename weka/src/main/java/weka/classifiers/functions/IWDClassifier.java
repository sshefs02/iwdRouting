/**
 * 
 */
package weka.classifiers.functions;

import java.io.Serializable;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Vector;

import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import weka.classifiers.RandomizableClassifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @author smriti srivastava
 *
 */
public class IWDClassifier extends RandomizableClassifier {

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
	
	// To store the final weightValues of the Neural Net
	private Vector <Vector <Double >> weightValues;
	
	/**
	 * soilValues maps a pair (j,i), where i is the index of current
	 * weight w_i, and j is the index of selected value of w_i, to all the
	 * paths which emanate from node (j,i) in the IWD graph
	 */
	private Map < Pair, Vector<Edge>> soilValues;
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
		numNodesInHiddenLayer = 10;
		numWeights = 0;
		precision = 0;
		numEpochs = 1;
		numIterations = 1;
		
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
		soilValues = new HashMap <Pair, Vector<Edge>>();
		
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
		while(numEpochs>0) {
			for (Instance instance : instances) {
				for (int i=0; i<numIterations; i++) {
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
		
		initializeWeights();
		initializePathSoil();
		placeIWDs();
	}

	private void initializeWeights() {
		// TODO add the biases if needed! 
		int valueRanges = 601 ;
		
		// 2D matrix with size = valueRanges * numWeights;
		weightValues.setSize(valueRanges);
		
		for ( int i = 0 ; i < valueRanges ; i++ ) {
			Vector <Double> values = new Vector<Double>();
			values.setSize(numWeights);
			weightValues.set(i, values);
		}
		
		int numOfRows = valueRanges;
		int numOfCols = numWeights;
		
		for ( int col = 0 ; col < numOfRows ; col++ ) { 
			double genValue = -30 ; 
			for ( int row = 0 ; row < numOfCols ; row++ ) {
				weightValues.get(col).set(row, genValue) ; 
				genValue += 0.1 ;
			}
		}
		
		return;
	}
	
	private void initializePathSoil() {
		int valueRanges = 601 ;
		
		int numOfRows = valueRanges;
		int numOfCols = numWeights;
		
		// TODO Selection of the init soil value.
		double initSoil = 500;
		
		for ( int col = 0 ; col < numOfCols ; col++ ) { 
			for ( int row = 0 ; row < numOfRows ; row++ ) {
				Edge edgeRowColumn = new Edge(initSoil, Double.POSITIVE_INFINITY);
				Vector<Edge> currSoilVals = new Vector<Edge>(Collections.nCopies(valueRanges, edgeRowColumn));
				Pair currPair = new Pair ( row, col );
				soilValues.put( currPair , currSoilVals ) ;
			}
		}
		
		return;
	}

	private void placeIWDs() {
		//One IWD is placed at all probable values of the first weight
		int numIwds = 601;
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
		
		DenseMatrix firstWeightMatrix = new DenseMatrix(numAttributes , numNodesInHiddenLayer);
		double instanceArray[] = new double[numAttributes];
		System.arraycopy(instance.toDoubleArray(), 0, instanceArray, 0, numAttributes);
		DenseVector inputVector = new DenseVector(instanceArray, false);
		
		int i = 0 ;
		for ( int r = 0 ; r < numAttributes ; r++ ) {
			for ( int c = 0 ; c < numNodesInHiddenLayer ; c++ ) {
				firstWeightMatrix.set(r, c, weightValues.get(iWDPath[i]).get(i));
				i++;
			}
		}
		
		//System.out.println(firstWeightMatrix.toString());
		
		DenseVector hiddenLayerOutput = new DenseVector(numNodesInHiddenLayer) ;
		firstWeightMatrix.transMult(inputVector, hiddenLayerOutput);
		
		//System.out.println(hiddenLayerOutput.toString());
		
		double output = 0 ; 
		int ctr = 0 ;
		for ( ; i < numWeights ; i++ ) {
			//System.out.println(hiddenLayerOutput.get(ctr) + "\t"+ weightValues.get(iWDPath[i]).get(i));
			output += hiddenLayerOutput.get(ctr) * weightValues.get(iWDPath[i]).get(i);
			ctr++ ; 
		}
		
		return output ; 
	}

	private double makeIWDJourney(Instance instance) {
		double leastError = Double.POSITIVE_INFINITY;
		//System.out.println(IWDs.size());
		int numIWDs = IWDs.size();
		for (int i=0 ; i<numIWDs; i++) {
			//System.out.println("Entered Loop");
			IWD iwd = IWDs.get(i);
			int IWDPath[] = iwd.traversePaths();
			double prediction = getPrediction(instance, IWDPath);
			double currentIWDPathError = calculateError(instance, prediction);
			setHUDs(IWDPath, currentIWDPathError);
			//System.out.println(prediction);
			
			/*
			 * Since quality is inverse of error, selecting path
			 * with least error is equivalent to selecting path with
			 * greatest quality
			 */
			if (currentIWDPathError < leastError) {
				leastError  = currentIWDPathError;
				currentIterationBestPathIndices = IWDPath;
				bestIWD = iwd;
			}
		}
		return leastError;
	}

	private void setHUDs(int IWDPath[], double error) {

		for (int i=0;i<IWDPath.length-1;i++) {
			Pair row_col = new Pair(IWDPath[i], i) ;
			try {
				if ( soilValues.get(row_col).get(IWDPath[i+1]).HUD > error) { 
					soilValues.get(row_col).get(IWDPath[i+1]).HUD  = error ; 
				}
			}
			catch (NullPointerException ex) {
				System.out.println(i+"\t"+IWDPath[i]);
				System.out.println(soilValues.get(new Pair(0,0)));
				ex.printStackTrace();
				System.exit(1);
			}
		}
	}

	
	private void updateGlobalSoil() {
		for (int i=0;i<numWeights-1;i++) {
			Pair nodePosition = new Pair(i, currentIterationBestPathIndices[i]);
			double currentSoil = soilValues.get(nodePosition).
					get(currentIterationBestPathIndices[i+1]).soilValue;
			double newSoil = (1 + rho_IWD) * currentSoil
					- rho_IWD * (1/(numWeights - 1))*bestIWD.soil;
			soilValues.get(nodePosition).get(bestPathIndices[i+1]).soilValue=newSoil;
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

	class IWD implements Serializable{
		
		/**
		 * 
		 */
		private static final long serialVersionUID = -8492864630943739397L;
		private double soil;
		private double velocity;
		private Pair currentPosition;
		
		private int selectedWeightIndices[];
		
		IWD(Pair position) {
			currentPosition = position;
			velocity = 200;
			soil = 10000;
			
			selectedWeightIndices = new int[numWeights];
			selectedWeightIndices[0] = currentPosition.y;
		}
		
		public int[] traversePaths() {
			while (currentPosition.y < numWeights) {
				Vector <Edge> soilValuesOfPaths = soilValues.get(currentPosition);
				try {
					int index = getNextIWDJump(soilValuesOfPaths);
					double soilOfSelectedPath = soilValuesOfPaths.get(index).soilValue;
					velocity += a_v/(b_v + c_v * soilOfSelectedPath*soilOfSelectedPath);

					double time = soilValuesOfPaths.get(index).HUD/velocity;
					double delSoil = a_s/(b_s + c_s * time*time);
					soilOfSelectedPath = (1 - rho_n)*soilOfSelectedPath
							- rho_n * delSoil;
					soil += delSoil;

					currentPosition.x = index;
					currentPosition.y++;
					if (currentPosition.y < numWeights)
						selectedWeightIndices[currentPosition.y] = currentPosition.x;
				}
				catch (NullPointerException ex) {
					System.out.println(currentPosition.x+" "+currentPosition.y);
					System.exit(1);
				}

			}
			return selectedWeightIndices;
		}
		
		// Return g_soil(i,j) while standing at node i and comparing with nodes of next layer. 
		private Vector<Edge> getGSoil(Vector<Edge> soil_ij) {
			soil_ij = new Vector<Edge>(soil_ij);
			//Calculating minimum ij
			double minSoil = Double.POSITIVE_INFINITY;
			for(Edge ii : soil_ij) {
					if( Double.compare( ii.soilValue , minSoil ) < 0 ) {
						minSoil = ii.soilValue ;
					}
			}
			if ( minSoil < 0 ) {
				//All soil values become (soil-minSoil)
				for (int i=0;i<soil_ij.size();i++) {
					Double ii = soil_ij.get(i).soilValue;
					ii = ii - minSoil;
					soil_ij.get(i).soilValue = ii;
				}
			}
			return soil_ij;
		}


		// Return f_soil(i,j) while standing at node i and comparing with nodes of next layer. 
		private Vector<Edge> getFSoil(Vector<Edge> g_soil_ij) {
			double epsilon_s = 0.01;
			// Lesson learnt : for each makes a copy
			for(int i=0;i<g_soil_ij.size();i++) {
				//Calculating f_soil_ij from g_soil_ij
				Double ii = g_soil_ij.get(i).soilValue;
				ii = 1 / ( epsilon_s + ii); 
				g_soil_ij.get(i).soilValue = ii;
			}
			return g_soil_ij;
		}
		
		//Return best next jump. Should calculate P(soil(i,j)) but not needed.
		public int getNextIWDJump(Vector<Edge> soil_ij) {
			Vector<Edge> f_soil_ij = getFSoil(getGSoil(soil_ij));
			double sigma_f_soil = 0;
			double cumulative_probability = 0;
			int i;

			double random_number = random.nextDouble();

			for (i=0;i<f_soil_ij.size();i++) {
				sigma_f_soil += f_soil_ij.get(i).soilValue;
			}
			
			for (i=0;i<f_soil_ij.size();i++) {
				cumulative_probability += f_soil_ij.get(i).soilValue/sigma_f_soil;
				if (random_number <  cumulative_probability) break;
			}
				
			return i;
		}
	}

	class Edge implements Serializable {
		
		/**
		 * 
		 */
		private static final long serialVersionUID = 2109248589392239227L;
		public double soilValue;
		public double HUD;
		
		public Edge(double soilValue, double HUD) {
			this.soilValue = soilValue;
			this.HUD = HUD;
		}
	}

	class Pair implements Serializable{
		/**
		 * 
		 */
		private static final long serialVersionUID = -3391667488014405636L;
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
