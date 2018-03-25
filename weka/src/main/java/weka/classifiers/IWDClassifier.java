/**
 * 
 */
package weka.classifiers;

import java.util.Map;
import java.util.Random;
import java.util.Vector;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;

/**
 * @author smriti srivastava
 *
 */
public class IWDClassifier implements Classifier, Randomizable {

	private Instances instances;
	private int numAttributes;
	private int numClasses;
	private int numNodesInHiddenLayer;
	private int numWeights;
	
	/** Random number generator */
	private Random random;
	private int randomSeed;
	private Vector <IWD> IWDs;
	private Vector <Vector <Double >> weightValues;
	
	/**
	 * soilValues maps a pair (j,i), where i is the index of current
	 * weight w_i, and j is the index of selected value of w_i, to all the
	 * paths which emanate from node (j,i) in the IWD graph
	 */
	private Map < Pair, Vector<Double>> soilValues;
	
	/**
	 * Constructor
	 */
	
	public IWDClassifier() {
		instances = null;
		numAttributes = 0;
		numClasses = 0;
		numNodesInHiddenLayer = 0;
		numWeights = 0;
		
		random = null;
		randomSeed = 0;
		
		IWDs = new Vector<IWD>();
		weightValues = new Vector <Vector < Double> >();
		soilValues = new HashMap <Pair, Vector<Double>>();
	}
	/* (non-Javadoc)
	 * @see weka.classifiers.Classifier#buildClassifier(weka.core.Instances)
	 */
	@Override
	public void buildClassifier(Instances data) throws Exception {
		initializeClassifier(data);
		for (Instance instance : instances) {
			makeIWDJourney();
			calculateError(instance);
		}
		
	}

	private void calculateError(Instance instance) {
		// TODO Auto-generated method stub
		
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
		
		placeIWDs();
	}

	private void placeIWDs() {
		//TODO
	}
	
	/* (non-Javadoc)
	 * @see weka.classifiers.Classifier#classifyInstance(weka.core.Instance)
	 */
	@Override
	public double classifyInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return 0;
	}


	/* (non-Javadoc)
	 * @see weka.classifiers.Classifier#getCapabilities()
	 */
	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}
	
	private double makeIWDJourney() {
		for (IWD i : IWDs) {
			Pair curPos = i.getCurrentPosition();
			int index_i = i.getNextIWDJump(soilValues.get(curPos));
			Pair nextPos = new Pair(index_i, curPos.y+1);
			i.setCurrentPosition(nextPos);
			
			//update
			//save weight values
			
		}

		//Calculate best HUD ( least error )
		//Global Soil Updation
		
		return 0;
	}
	
	
	/**
	 * @return the randomSeed
	 */
	@Override
	public int getSeed() {
		return randomSeed;
	}
	/**
	 * @param randomSeed the randomSeed to set
	 */
	@Override
	public void setSeed(int randomSeed) {
		this.randomSeed = randomSeed;
	}


	class IWD {
		
		private double Soil;
		private double Velocity;
		private Pair currentPosition;
		
		// Return g_soil(i,j) while standing at node i and comparing with nodes of next layer. 
		private Vector<Double> getGSoil(Vector<Double> soil_ij) {
			//Calculating minimum ij
			double minSoil = Double.POSITIVE_INFINITY;
			for(Double ii : soil_ij) {
					if( Double.compare( ii , minSoil ) < 0 ) {
						minSoil = ii ;
					}
			}
			if ( minSoil < 0 ) {
				//All soil values become (soil-minSoil)
				for(Double ii : soil_ij) {
					ii = ii - minSoil;
				}
			}
			return soil_ij;
		}


		// Return f_soil(i,j) while standing at node i and comparing with nodes of next layer. 
		private Vector<Double> getFSoil(Vector<Double> g_soil_ij) {
			double epsilon_s = Double.POSITIVE_INFINITY;
			for(Double ii : g_soil_ij) {
				//Calculating f_soil_ij from g_soil_ij
				ii = 1 / ( epsilon_s + ii); 
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
				index++;
				if( ii > maxF) {
					maxF = ii;
					indexMax = index; 
				}
			}
			return indexMax;
		}

		
		public double getSoil() {
			return Soil;
		}


		public void setSoil(double soil) {
			Soil = soil;
		}


		public double getVelocity() {
			return Velocity;
		}


		public void setVelocity(double velocity) {
			Velocity = velocity;
		}


		public Pair getCurrentPosition() {
			return currentPosition;
		}


		public void setCurrentPosition(Pair currentPosition) {
			this.currentPosition = currentPosition;
		}
	}

	class Pair {
		int x;
		int y;
		
		Pair(int x, int y) { this.x = x; this.y = y;}
	}
	
}
