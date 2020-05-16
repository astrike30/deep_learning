import numpy as np
import scipy.special


class NeuralNetModel:
    def __init__(self, noInputs, noHiddens, noOutputs, lr):

        self.noInputs = noInputs
        self.noHiddens = noHiddens
        self.noOutputs = noOutputs
        self.lr = lr

        self.weights = []
        self.biases = []

        self.weights.append(
            np.random.normal(0.0, pow(noHiddens[0], -0.50), (noHiddens[0], noInputs))
        )
        self.biases.append(np.zeros((noHiddens[0], 1)))
        for i in range(1, len(noHiddens)):
            self.weights.append(
                np.random.normal(0.0, pow(noHiddens[i], -0.50), (noHiddens[i], noHiddens[i - 1]))
            )
            self.biases.append(np.zeros((noHiddens[i], 1)))

        self.weights.append(
            np.random.normal(0.0, pow(noOutputs, -0.50), (noOutputs, noHiddens[-1]))
        )
        self.biases.append(np.zeros((noOutputs, 1)))

    def activation_function(self, x):
        return scipy.special.expit(x)

    def query(self, inputs):
        if len(inputs[0]) != self.noInputs:
            raise ValueError("Invalid Input Shape")

        inputs = np.array(inputs, ndmin=2).T
        current = self.weights[0].dot(inputs)
        current = np.add(current, self.biases[0])
        current = self.activation_function(current)
        for i in range(1, len(self.noHiddens) + 1):
            current = self.weights[i].dot(current)
            current = np.add(current, self.biases[i])
            current = self.activation_function(current)

        return current.transpose()

    def train(self, inputs, targets):
        # inputs must be a Python list with dimensions (1, noInputs)
        # this is ensured in the query methods

        # targets must have dimensions (1, noOutputs)
        # we check one here
        if len(targets[0]) != self.noOutputs:
            raise ValueError("Invalid Output Shape")

        targets = np.array(targets).T
        inputs = np.array(inputs, ndmin=2).T

        current = self.weights[0].dot(inputs)
        current = np.add(current, self.biases[0])
        current = self.activation_function(current)
        outputs = [inputs, current]
        for i in range(1, len(self.noHiddens) + 1):
            current = self.weights[i].dot(current)  # + self.biases[i]
            current = np.add(current, self.biases[i])
            current = self.activation_function(current)
            outputs.append(current)

        output_error = targets - current
        for i in reversed(range(0, len(self.noHiddens) + 1)):
            self.weights[i] += self.lr * np.dot(
                (output_error * outputs[i + 1] * (1.0 - outputs[i + 1])), np.transpose(outputs[i])
            )
            self.biases[i] += self.lr * outputs[i + 1] * (1.0 - outputs[i + 1])
            output_error = np.dot(self.weights[i].T, output_error)

        return self.weights


if __name__ == "__main__":
    nn = NeuralNetModel(1, [4, 4, 4], 2, 1)

    print(nn.query([[10], [10], [10]]))
    (nn.train([[10], [10], [10]], [[1, 1], [0.5, 0.5], [0.1, 0.1]]))
    print(nn.query([[10], [10], [10]]))
