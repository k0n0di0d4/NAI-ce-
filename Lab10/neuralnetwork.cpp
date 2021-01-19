#include<vector>
#include<iostream>
#include<cassert>
#include<cmath>

class Neuron;
typedef std::vector<Neuron> Layer;
struct Connection {
	double weight;
	double deltaWeight;
};


//class defining the Neuron
class Neuron {
public:
	Neuron(unsigned numOutputs, unsigned myIndex);
	void setOutputValue(double value) { m_outputValue = value; }
	double getOutputValue(void) const { return m_outputValue; }
	void feedForward(const Layer &prevLayer);
	void calcOutputGradients(double targetValue);
	void calcHiddenGradients(const Layer& nextLayer);
	void updateInputWeights(Layer &prevLayer);

private:
	static double randomWeight(void) {
		return rand() / double(RAND_MAX);
	}
	static double transferFunction(double x);
	static double transferFunctionDerivative(double x);
	double m_outputValue;
	std::vector<Connection> m_outputWeights;
	unsigned m_myIndex;
	double m_gradient;
	double sumDOW(const Layer &nextLayer) const;
	static double eta; // [0.0..1.0] overall net training rate
	static double alpha; // [0.0..n] multiplier of last weight change (momentum)
};

double Neuron::eta = 0.15; // overall net learning rate
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight 


//the constructor
Neuron::Neuron(unsigned numOutputs, unsigned myIndex) {
	for (unsigned c = 0; c < numOutputs; ++c) {
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
	}
	m_myIndex = myIndex;
}


double Neuron::transferFunction(double x) {
	// tanh - output range [-1.0, 1.0]
	return tanh(x);
}


double Neuron::transferFunctionDerivative(double x) {
	//tanh derivative
	return 1.0 - x * x;
}


void Neuron::feedForward(const Layer &prevLayer) {
	double sum = 0.0;

	// Sum the previous layer's outputs (which are our inputs)
	// include the bias node from the previous layer

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		sum += prevLayer[n].getOutputValue() * prevLayer[n].m_outputWeights[m_myIndex].weight;
	}
	m_outputValue = Neuron::transferFunction(sum);
}


void Neuron::calcOutputGradients(double targetValue) {
	double delta = targetValue - m_outputValue;
	m_gradient = delta * Neuron::transferFunctionDerivative(m_outputValue);
}


void Neuron::calcHiddenGradients(const Layer& nextLayer) {
	double dow = sumDOW(nextLayer);
	m_gradient = dow * Neuron::transferFunctionDerivative(m_outputValue);
}


double Neuron::sumDOW(const Layer& nextLayer) const{
	double sum = 0.0;
	
	//Sum our contributions of the errors at the nodes we feed

	for (unsigned n = 0; n < nextLayer.size(); ++n) {
		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}
	return sum;
}


void Neuron::updateInputWeights(Layer &prevLayer) {
	//The weights to be updated are in the Connection container
	//in the neurons in the preceding layer

	for (unsigned n = 0; n < prevLayer.size(); ++n) {
		Neuron& neuron = prevLayer[n];
		double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

		double newDeltaWeight =
			//Individual input, magnified by the gradient and train rate:
			eta
			* neuron.getOutputValue()
			* m_gradient
			//Also add momentum = a fraction of the previous delta weight
			+ alpha // alpha = momentum rate which is a multiplier of the old changing weight from the last sample
			* oldDeltaWeight;
		
		neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;

	}
}


//class defining the Neural Network
class Net {
public:
	Net(const std::vector<unsigned> &topology);
	void feedForward(const std::vector<double> &inputValues);
	void backPropagation(const std::vector<double> &targetValues);
	void getResults(std::vector<double> &resultValues) const;
	double getRecentAverageError(void) const { return m_recentAverageError; }
private:
	std::vector<Layer> m_layers;
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};

double Net::m_recentAverageSmoothingFactor = 100.0;

//topology is how neurons connect between each other
Net::Net(const std::vector<unsigned> &topology) {
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers;  ++layerNum) {
		m_layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];
		//We have made a new Layer, now fill it with neurons, and
		// add a bias neuron to the layer:
		for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum) {
			m_layers.back().push_back(Neuron(numOutputs, neuronNum));
			std::cout << "Made a Neuron!" << std::endl;
		}
	}
}


void Net::feedForward(const std::vector<double> &inputValues) {
	assert(inputValues.size() == m_layers[0].size() - 1);

	//Assign (latch) the input values into the input neurons
	for (unsigned i = 0; i < inputValues.size(); ++i) {
		//m_layers[0][i].setOutputValue(inputValues[i]);
	}

	//Forward Propagate
	for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum) {
		Layer &prevLayer = m_layers[layerNum - 1]; //reference to the previous layer
		for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n) { // -1 because we don't count the bias neuron
			m_layers[layerNum][n].feedForward(prevLayer);
		}
	}
}


void Net::backPropagation(const std::vector<double>& targetValues) {
	// Calculate overall net error (The good old RMS(Root Mean Square) of output neuron errors)

	Layer& outputLayer = m_layers.back();
	m_error = 0.0;

	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) {
		double delta = targetValues[n] - outputLayer[n].getOutputValue();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1; //get average error squared
	m_error = sqrt(m_error); //RMS

	//Implement a recent average measurement:
	//which is an Error indication of how well the net,
	//has been doing over the last several dozen training samples
	m_recentAverageError = (m_recentAverageError * m_recentAverageSmoothingFactor + m_error) / (m_recentAverageSmoothingFactor + 1.0);
	
	//Calculate output layer gradients
	for (unsigned n = 0; n < outputLayer.size() - 1; ++n) { //of course it's (-1) because we don't count the bias neuron
		outputLayer[n].calcOutputGradients(targetValues[n]);
	}

	//Calculate gradients on hidden layers
	for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum) {
		Layer& hiddenLayer = m_layers[layerNum];
		Layer& nextLayer = m_layers[layerNum + 1];
		for (unsigned n = 0; n < hiddenLayer.size(); ++n) {
			hiddenLayer[n].calcHiddenGradients(nextLayer);
		}
	}

	//For all layers from outputs to first hidden layer,
	//update connection weights
	for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum) {
		Layer& layer = m_layers[layerNum];
		Layer& prevLayer = m_layers[layerNum - 1];

		for (unsigned n = 0; n < layer.size() - 1; ++n) {
			layer[n].updateInputWeights(prevLayer);
		}
	}

}


void Net::getResults(std::vector<double> &resultValues) const{
	resultValues.clear();
	for (unsigned n = 0; n < m_layers.back().size(); ++n) {
		resultValues.push_back(m_layers.back()[n].getOutputValue());
	}
}


void showVectorValues(std::string label, std::vector<double> v) {
	std::cout << label << " ";
	for (unsigned i = 0; i < v.size(); ++i) {
		std::cout << v[i] << " ";
	}
	std::cout << std::endl;
}


int main() {
	std::vector<unsigned> topology;
	//generating three layers
	for (int i = 1; i <= 3; i++) {
		topology.push_back(i);
	}
	Net myNet(topology);

	std::vector<double> inputValues;
	myNet.feedForward(inputValues);
	
	std::vector<double> targetValues;
	myNet.backPropagation(targetValues);
	
	std::vector<double> resultValues;
	myNet.getResults(resultValues);
}