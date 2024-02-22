from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from nn import NeuralNetwork


def loadDigitData():
    from sklearn.datasets import load_digits
    data = load_digits()
    inputs = data.images
    outputs = data['target']
    outputNames = data['target_names']
     
    # shuffle the original data
    noData = len(inputs)
    permutation = np.random.permutation(noData)
    inputs = inputs[permutation]
    outputs = outputs[permutation]

    return inputs, outputs, outputNames


def splitData(inputs, outputs):
    np.random.seed(5)
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace = False)
    testSample = [i for i in indexes  if not i in trainSample]

    trainInputs = [inputs[i] for i in trainSample]
    trainOutputs = [outputs[i] for i in trainSample]
    testInputs = [inputs[i] for i in testSample]
    testOutputs = [outputs[i] for i in testSample]
    
    return trainInputs, trainOutputs, testInputs, testOutputs


def normalisation(trainData, testData):
    scaler = StandardScaler()

    if not isinstance(trainData[0], list):
        #encode each sample into a list
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]

        scaler.fit(trainData)  #  fit only on training data
        normalisedTrainData = scaler.transform(trainData) # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data

        #decode from list to raw values
        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        scaler.fit(trainData)  #  fit only on training data
        normalisedTrainData = scaler.transform(trainData) # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data
    
    return normalisedTrainData, normalisedTestData


def training(classifier, trainInputs, trainOutputs):
    # identify (by training) the classification model
    classifier.fit(trainInputs, trainOutputs)


def classification(classifier, testInputs):
    # makes predictions for test data 
    computedTestOutputs = classifier.predict(testInputs)

    return computedTestOutputs


def data2FeaturesMoreClasses(inputs, outputs, outputNames):
    labels = set(outputs)
    noData = len(inputs)
    for crtLabel in labels:
        x = [inputs[i][0] for i in range(noData) if outputs[i] == crtLabel]
        y = [inputs[i][1] for i in range(noData) if outputs[i] == crtLabel]
        plt.scatter(x, y, label = outputNames[crtLabel])
    plt.xlabel('feature1')
    plt.ylabel('feature2')
    plt.legend()
    plt.show() 


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plotConfusionMatrix(cm, classNames, title):
    from sklearn.metrics import confusion_matrix
    import itertools 

    classes = classNames
    plt.figure()
    plt.imshow(cm, interpolation = 'nearest', cmap = 'Blues')
    plt.title('Confusion Matrix ' + title)
    plt.colorbar()
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)

    text_format = 'd'
    thresh = cm.max() / 2.
    for row, column in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(column, row, format(cm[row, column], text_format),
                horizontalalignment = 'center',
                color = 'white' if cm[row, column] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    plt.show()


def evalMultiClass(realLabels, computedLabels, labelNames):
    from sklearn.metrics import confusion_matrix

    confMatrix = confusion_matrix(realLabels, computedLabels)
    acc = sum([confMatrix[i][i] for i in range(len(labelNames))]) / len(realLabels)
    precision = {}
    recall = {}
    for i in range(len(labelNames)):
        precision[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[j][i] for j in range(len(labelNames))])
        recall[labelNames[i]] = confMatrix[i][i] / sum([confMatrix[i][j] for j in range(len(labelNames))])
    return acc, precision, recall, confMatrix


def flatten(mat):
    x = []
    for line in mat:
        for el in line:
            x.append(el)
    return x 


def digitProblem():
    inputs, outputs, outputNames = loadDigitData()
    trainInputs, trainOutputs, testInputs, testOutputs = splitData(inputs, outputs)
    plt.hist(trainOutputs, rwidth = 0.8)
    plt.xticks(np.arange(len(outputNames)), outputNames)
    plt.show()

    trainInputsFlatten = [flatten(el) for el in trainInputs]
    testInputsFlatten = [flatten(el) for el in testInputs]
    trainInputsNormalised, testInputsNormalised = normalisation(trainInputsFlatten, testInputsFlatten)
    classifier = NeuralNetwork([32, 16], sigmoid, 0.1, 100)

    training(classifier, trainInputsNormalised, trainOutputs)
    predictedLabels = classification(classifier, testInputsNormalised)

    lossPerEpochs = classifier.getLossPerEpochs()
    plt.plot(lossPerEpochs)

    acc, prec, recall, cm = evalMultiClass(np.array(testOutputs), predictedLabels, outputNames)

    plotConfusionMatrix(cm, outputNames, "digit classification")
    print('acc: ', acc)
    print('precision: ', prec)
    print('recall: ', recall)

    # plot first 50 test images and their real and computed labels
    n = 10
    m = 5
    fig, axes = plt.subplots(n, m, figsize = (7, 7))
    fig.tight_layout() 
    for i in range(0, n):
        for j in range(0, m):
            axes[i][j].imshow(testInputs[m * i + j])
            if (testOutputs[m * i + j] == predictedLabels[m * i + j]):
                font = 'normal'
            else:
                font = 'bold'
            axes[i][j].set_title('real ' + str(testOutputs[m * i + j]) + '\npredicted ' + str(predictedLabels[m * i + j]), fontweight=font)
            axes[i][j].set_axis_off()
            
    plt.show()


def main():
    digitProblem()

main()