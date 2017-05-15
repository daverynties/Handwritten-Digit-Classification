import time
import random
import numpy as np
from sklearn.preprocessing import scale
from collections import Counter
from sklearn.metrics import confusion_matrix
import warnings
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

class NeuralNetwork(object):

    def __init__(self, input, hidden, output, iterations=50, learning_rate=0.01):

        # initialize parameters
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.input = input + 1  # add 1 for bias node
        self.hidden = hidden
        self.output = output

        # set up array of 1s for activations
        self.activationInputs = np.ones(self.input)
        self.activationHidden = np.ones(self.hidden)
        self.activationOutputs = np.ones(self.output)

        # create randomized weights
        input_range = 1.0 / self.input ** (1 / 2)
        self.weightedInputs = np.random.normal(loc=0, scale=input_range, size=(self.input, self.hidden))
        self.weightedOutputs = np.random.uniform(size=(self.hidden, self.output)) / np.sqrt(self.hidden)

        # create arrays of 0 for changes
        self.changeInputs = np.zeros((self.input, self.hidden))
        self.changeOutputs = np.zeros((self.hidden, self.output))

    def FeedForward(self, inputs):
        # Loop through all nodes in hidden and calculate sum of input_layer * weights.
        # Output of each node is the sigmoid of the sum of all inputs.
        # Pass values to next layer

        # input activations
        self.activationInputs[0:self.input - 1] = inputs

        # hidden activations
        # initially used sigmoid for the activation function on the hidden layer, but switched to tanh
        # which netted a 6% increase in classification accuracy
        sum = np.dot(self.weightedInputs.T, self.activationInputs)
        self.activationHidden = np.tanh(sum)

        # output activations
        sum = np.dot(self.weightedOutputs.T, self.activationHidden)
        self.activationOutputs = 1 / (1 + np.exp(-sum))

        return self.activationOutputs

    def BackPropagate(self, targets):

        # Calculate output error (dSigmoid -> weight change direction)
        output_deltas = self.activationOutputs * (1.0 - self.activationOutputs) * -(targets - self.activationOutputs)

        # Calculate hidden error (dTheta -> weight change direction)
        error = np.dot(self.weightedOutputs, output_deltas)
        hidden_deltas = (1 - (self.activationHidden * self.activationHidden)) * error

        # Update weight values on hidden to output
        change = output_deltas * np.reshape(self.activationHidden, (self.activationHidden.shape[0], 1))
        self.weightedOutputs -= (self.learning_rate * change)
        self.changeOutputs = change

        # Update weight values on input to hidden
        change = hidden_deltas * np.reshape(self.activationInputs, (self.activationInputs.shape[0], 1))
        self.weightedInputs -= (self.learning_rate * change)
        self.changeInputs = change

        # Calculate error
        error = sum(targets - self.activationOutputs) ** 2

        return error


    def Fit(self, patterns):

        num_example = np.shape(patterns)[0]

        for i in range(self.iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.FeedForward(inputs)
                error += self.BackPropagate(targets)
                print error

            if i % 1 == 0:
                error = error / num_example
                print '%-.5f' % error


    def Predict(self, testData, expected):

        predictions = []
        incorrectValues = []
        conMatrixActual = []
        conMatrixPredicted = []

        i = 0

        for digitMatrix in testData:
            outputPredictionArray = self.FeedForward(digitMatrix)
            maxOutputIndex = np.argmax(outputPredictionArray, 0)
            if maxOutputIndex == expected[i]:
                predictions.append('Correct')
            else:
                predictions.append('Incorrect')
                incorrectValues.append((maxOutputIndex, expected))
            print 'Predicted {} --> Expected {}'.format(maxOutputIndex, int(expected[i]))
            conMatrixActual.append(expected[i])
            conMatrixPredicted.append(maxOutputIndex)
            i += 1
        conMatrix = confusion_matrix(conMatrixActual, conMatrixPredicted)

        ax = sns.heatmap(conMatrix)
        plt.savefig('conMatrix.png', dpi=100)
        plt.show()

        print
        print conMatrix
        return predictions


if __name__ == '__main__':

    start = time.time()

    data = np.loadtxt('trainingSet.csv', delimiter=',')
    testData = np.loadtxt('testSet.csv', delimiter=',')
    test_raw = data[:, -1]
    testData_raw = testData[:, -1]
    outputMatrix = []

    for value in test_raw:
        outputMatrixArray = np.zeros(10)
        np.put(outputMatrixArray, [value], [1])
        outputMatrix.append(outputMatrixArray)
        myarray = np.asarray(outputMatrix)

    dataNew = scale(data)
    dataTest = scale(testData)
    Hsub = dataNew[:, 0:-1]
    Tsub = dataTest[:, 0:-1]

    normalizedIOData = []

    for i in range(dataNew.shape[0]):
        tupledata = list((Hsub[i, :].tolist(), myarray[i].tolist()))
        normalizedIOData.append(tupledata)

    NN = NeuralNetwork(64, 50, 10, iterations=50, learning_rate=0.3)
    NN.Fit(normalizedIOData)

    end = time.time()

    predictedValueList = Counter(NN.Predict(Tsub, testData_raw))

    correct = predictedValueList.get('Correct')
    incorrect = predictedValueList.get('Incorrect')

    totalValues = correct + incorrect

    print '''\nTotal Correct Predictions: {}\nTotal Incorrect Predictions: {}\nAccuracy: {}'''.format\
        (correct, incorrect, float(correct) / totalValues)
    print '\nExecution Time: {}'.format(end - start)
