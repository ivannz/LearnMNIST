loadMNISTData <- function(dataFilename, labelFilename){
  # Loads the data from the MNIST dataset. The data can be downloaded from: 
  # http://yann.lecun.com/exdb/mnist/
  #
  # Args:
  # dataFilename: specifies the name of the file containing the data
  # labelFilename: specifies the name of the file containing the labels
  
  # Returns:
  # The data and the labels combined in a list data structure
  
  dataFileIn <- if( !gz ) file(dataFilename, "rb") else gzfile(dataFilename, "rb")
  
  #read the first unsigned byte
  dummyByte = readBin(dataFileIn, integer(), n = 1, size = 4, endian = "big")
  if (dummyByte != 2051) {
    stop("Error reading file.")
  }
  
  #read the number of data
  nData = readBin(dataFileIn, integer(), n = 1, size = 4, endian = "big")
  x <- readBin(dataFileIn, integer(), n = 1, size = 4, endian = "big")
  y <- readBin(dataFileIn, integer(), n = 1, size = 4, endian = "big")
  
  dataSize = x*y
  
  data <- matrix(data = 0, nrow = nData, ncol = dataSize)
  
  for (i in 1:nData){
    data[i,] <- readBin(dataFileIn, integer(), n = dataSize, size = 1, endian = "big", signed = FALSE)
  }
  
  close(dataFileIn)
  
  #read the labels
  #open MNIST data file; 
  dataFileIn <- if( !gz ) file(labelFilename, "rb") else gzfile(labelFilename, "rb")
  
  #read the first unsigned byte
  dummy_byte = readBin(dataFileIn, integer(), n = 1, size = 4, endian = "big")
  if (dummy_byte != 2049) {
    stop("Error reading file")
  }
  
  #read the number of data
  nData = readBin(dataFileIn, integer(), n = 1, size = 4, endian = "big")
  
  dataSize = 1
  
  labels <- matrix(data = 0, nrow = nData, ncol = dataSize)
  
  for (i in 1:nData){
    labels[i] <- readBin(dataFileIn, integer(), n = 1, size = 1, endian = "big", signed = FALSE)
  }
  
  close(dataFileIn);
  
  return(list("data" = data, "labels" = labels))
}


