rm( list = ls( all.names = TRUE ) ) ; gc( )
# This sciprt file contains a frame for learning handwritten digitals from the MNIST dataset
setwd( "~/Github/LearnMNIST/" )
source( "load_data.R" )

# load training data from files
# data <- loadMNISTData("C:\\Users\\User\\YandexDisk\\teaching\\advanced_topics_in_machine_learning\\train-images.idx3-ubyte", "C:\\Users\\User\\YandexDisk\\teaching\\advanced_topics_in_machine_learning\\train-labels.idx1-ubyte", gzip = FALSE )
train <- loadMNISTData("./data/train-images.idx3-ubyte.gz", "./data/train-labels.idx1-ubyte.gz", gzip = TRUE )
trainLabels <- train$labels
trainData <- train$data

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

source( "metrics.R" )
#calculate the following error metric for each class obtained on the train data:
#Recall, precision, specificity, F-measure, FDR and ROC for each class separately. Use a package for ROC. 
metrics <- cbind( precision( trainLabels, predictedLabels ),
                  recall( trainLabels, predictedLabels ),
                  F_measure( trainLabels, predictedLabels ),
                  specificity( trainLabels, predictedLabels ),
                  fdr( trainLabels, predictedLabels ) )
names( dimnames( metrics ) ) <- c( "class", "metric" )
print( metrics )
roc_curve( trainLabels, predict_proba( classifier, trainData ) )


# test the model
# data <- loadMNISTData("C:\\Users\\User\\YandexDisk\\teaching\\advanced_topics_in_machine_learning\\t10k-images.idx3-ubyte", "C:\\Users\\User\\YandexDisk\\teaching\\advanced_topics_in_machine_learning\\t10k-labels.idx1-ubyte", gzip = FALSE )
data <- loadMNISTData("./data/t10k-images.idx3-ubyte.gz", "./data/t10k-labels.idx1-ubyte.gz", gzip = TRUE )
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
metrics <- cbind( precision( testLabels, predictedLabels ),
                  recall( testLabels, predictedLabels ),
                  F_measure( testLabels, predictedLabels ),
                  specificity( testLabels, predictedLabels ),
                  fdr( testLabels, predictedLabels ) )
names( dimnames( metrics ) ) <- c( "class", "metric" )
print( metrics )
roc_curve( testLabels, predict_proba( classifier, testData ) )

