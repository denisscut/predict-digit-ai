import numpy as np

class NeuralNetwork:
    def __init__(self, noHiddenNodes, activationFunction, learningRate, epochs):
        self.__noHiddenNodes = noHiddenNodes
        self.__epochs = epochs

        if type(noHiddenNodes) != list:
            self.__noHiddenNodes = [noHiddenNodes]
        self.__noHiddenLayers = len(self.__noHiddenNodes)

        self.__activationFunction = activationFunction
        self.__learningRate = learningRate
        self.__lossPerEpochs = []

    def __initNeuralNetwork(self, noInputNodes, noOutputNodes):
        self.__noInputNodes = noInputNodes
        self.__noOutputNodes = noOutputNodes
        self.__weight = []
        self.__weight.append(np.random.normal(0, 0.1, size=(self.__noInputNodes, self.__noHiddenNodes[0])))
        for i in range(self.__noHiddenLayers - 1):
            self.__weight.append(np.random.normal(0, 0.1, size=(self.__noHiddenNodes[i], self.__noHiddenNodes[i + 1])))
        self.__weight.append(np.random.normal(0, 0.1, size=(self.__noHiddenNodes[-1], self.__noOutputNodes)))

        self.__bias = []
        self.__bias.append(np.random.normal(0, 0.1, size=(1, self.__noHiddenNodes[0])))
        for i in range(self.__noHiddenLayers - 1):
            self.__bias.append(np.random.normal(0, 0.1, size=(1, self.__noHiddenNodes[i + 1])))
        self.__bias.append(np.random.normal(0, 0.1, size=(1, self.__noOutputNodes)))

    def __forwardPropagation(self, input):
        computedOutputs = []
        for i, (weight, bias) in enumerate(zip(self.__weight, self.__bias)):
            if i == 0:
                computedOutput = np.dot(input, weight) + bias
            else:
                computedOutput = np.dot(computedOutputs[-1], weight) + bias
            
            computedOutputs.append(self.__activationFunction(computedOutput))
        return computedOutputs

    def __backwardPropagation(self, input, output, computedOutputs):
        for i in range(len(self.__weight) - 1, -1, -1):
            if i == len(self.__weight) - 1:
                errorOutput = np.multiply(output - computedOutputs[i], (np.multiply(computedOutputs[i], (1 - computedOutputs[i]))))
            else:
                errorOutput = np.multiply(prevErrorOutput.dot(self.__weight[i + 1].T), np.multiply(computedOutputs[i], (1 - computedOutputs[i])))

            prevErrorOutput = errorOutput
            if i == 0:        
                self.__weight[i] += input.T.dot(errorOutput) * self.__learningRate
            else:
                self.__weight[i] += computedOutputs[i - 1].T.dot(errorOutput) * self.__learningRate
            self.__bias[i] += errorOutput * self.__learningRate


    def getLossPerEpochs(self):
        return self.__lossPerEpochs

    def __getLoss(self, outputs, computedOutputsLayer):
        loss = 0
        for output, computedOutputsLayer in zip(outputs, computedOutputsLayer):
            if self.__getLabel(output) != self.__getLabel(computedOutputsLayer):
                loss += 1
        return loss / len(outputs)

    def __getLabel(self, computedOutput):
        result = computedOutput.tolist()[0]
        i = result.index(max(result))
        return self.__indexToLabel[i]

    def __convertLabelsToOutputs(self, labels):
        setOfLabels = set(labels)
        self.__labelToIndex = {}
        self.__indexToLabel = {}

        i = 0
        for label in setOfLabels:
            self.__labelToIndex[label] = i
            self.__indexToLabel[i] = label
            i += 1

        numberOfLabels = len(setOfLabels)
        outputs = []
        for label in labels:
            i = self.__labelToIndex[label]
            output = [0 if j != i else 1 for j in range(numberOfLabels)]
            outputs.append(output)
        return outputs

    def fit(self, inputs, labels):
        outputs = self.__convertLabelsToOutputs(labels)
        self.__initNeuralNetwork(len(inputs[0]), len(outputs[0]))

        for epoch in range(self.__epochs):
            print("Epoch:", epoch, end=" ")
            computedOutputsLayer = []
            for input, output in zip(inputs, outputs):
                inputLayer = np.matrix(input)
                outputLayer = np.matrix(output)

                computedOutputs = self.__forwardPropagation(inputLayer)
                computedOutputsLayer.append(computedOutputs[-1])
                self.__backwardPropagation(inputLayer, outputLayer, computedOutputs)

            loss = self.__getLoss(np.matrix(outputs), computedOutputsLayer)
            print("Loss:", self.__getLoss(np.matrix(outputs), computedOutputsLayer))
            self.__lossPerEpochs.append(loss)

    def predict(self, inputs):
        labels = []
        for input in inputs:
            inputLayer = np.matrix(input)
            computedOutputs = self.__forwardPropagation(inputLayer)
            label = self.__getLabel(computedOutputs[-1])
            labels.append(label)
        return labels