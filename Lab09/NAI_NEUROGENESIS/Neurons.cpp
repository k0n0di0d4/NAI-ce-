#include <array>
#include<iostream>
#include<vector>

class Neuron
{
public:
	int parameter;

	double act_function(std::vector<double> weights, std::vector<bool> inputs) {
		int X = weights.size();
		double result = 0;
		switch (parameter) {
			case 0:
				for (int i = 0; i < X; i++) {
					if (inputs[i] == true) result += weights[i];
				}
				break;
			case 1:
				for(int i = 0; i < X; i++)
					if (inputs[i] == true) {
						if (weights[i] >= 0) {
							result += 1;
						}
					}
				break;
			case 2:
				for (int i = 0; i < X; i++)
					if (inputs[i] == true) {
						result += 1 / (1 / (1 + exp(-weights[i])));
					}
				break;
		}
		return result;
	}

	double TLU(double threshold, std::vector<double> weights, std::vector<bool> inputs, double (*function)(std::vector<double>, std::vector<bool>)) {
		if (function(weights, inputs) > threshold) return function(weights, inputs);
		else return 0;
	}
private:

};


std::vector<double> weights = {-100, 1, 2, 3.5 };
double threshold = 1.0;
std::vector<bool> inputs = {1, 1, 1, 0 };
int main() {
	Neuron neuron;
	neuron.parameter = 1;
	std::cout << neuron.TLU(threshold, weights, inputs, neuron.act_function(weights, inputs)) << std::endl;
	return 0;
}