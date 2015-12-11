# This sciprt file contains a frame for learning handwritten digitals from the MNIST dataset

# load training data from files
# data <- loadMNISTData("C:\\Users\\User\\YandexDisk\\teaching\\advanced_topics_in_machine_learning\\train-images.idx3-ubyte", "C:\\Users\\User\\YandexDisk\\teaching\\advanced_topics_in_machine_learning\\train-labels.idx1-ubyte", gz = FALSE )
data <- loadMNISTData("./data/train-images.idx3-ubyte", "./data/train-labels.idx1-ubyte", gz = TRUE )
trainLabels <- data$labels
trainData <- data$data

print(dim(trainData))
print(dim(trainLabels))
# trainingData should be 60000x786,  60000 data and 784 features (28x28), tha matrix trainData has 60000 rows and 784 columns
# trainingLabels should have 60000x1, one class label \in {0,1,...9} for each data.

#uncomment the following 3 lines to see the nth training example and its class label.
# n = 10;
# image( t(matrix(trainData[n, ], ncol=28, nrow=28)), Rowv=28, Colv=28, col = heat.colors(256),  margins=c(5,10))
# print("Class label:"); print(trainLabels[n])

# train a model
source( "logistic.R" )
classifier <- learnModel(data = trainData, labels = trainLabels)
predictedLabels <- testModel(classifier, trainData)

#calculate accuracy on training data
print("accuracy on training data:\t")
print(sum(predictedLabels == trainLabels)/length(trainLabels))

#calculate the following error metric for each class obtained on the train data:
#Recall, precision, specificity, F-measure, FDR and ROC for each class separately. Use a package for ROC. 


# test the model
# data <- loadMNISTData("C:\\Users\\User\\YandexDisk\\teaching\\advanced_topics_in_machine_learning\\t10k-images.idx3-ubyte", "C:\\Users\\User\\YandexDisk\\teaching\\advanced_topics_in_machine_learning\\t10k-labels.idx1-ubyte", gz = FALSE )
data <- loadMNISTData("./data/t10k-images.idx3-ubyte", "./data/t10k-labels.idx1-ubyte", gz = TRUE )
testLabels <- data$labels
testData <- data$data

print(dim(testData))
print(dim(testLabels))
#trainingData should be 10000x786,  10000 data and 784 features (28x28), tha matrix trainData has 10000 rows and 784 columns
#trainingLabels should have 10000x1, one class label \in {0,1,...9} for each data.

predictedLabels <- testModel(classifier, testData)

#calculate accuracy
print("accuracy on test data:\t")
print(sum(predictedLabels == testLabels)/length(testLabels))

#calculate the following error metric for each class obtained on the test data:
#Recall, precision, specificity, F-measure, FDR and ROC for each class separately. Use a package for ROC. 
