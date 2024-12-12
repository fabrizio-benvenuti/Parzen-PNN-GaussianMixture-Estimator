# Parzen-PNN-GaussianMixture-Estimator
## What it consists of:
- The project consists of estimating a two-dimensional PDF formed by a mixture of an odd number of Gaussians [1,3,5],\
   where the weights for composing the mixture and statistical parameters such as the mean and variance\
   are chosen to avoid PDFs with either excessively overlapping peaks or ones that are too distant from each other.

## How is it done:
- This is initially done by extracting a set of points from the PDF with intermediate cardinality\
    (approximately 100 points per Gaussian in the mixture, increasing as its mixture weight grows).
- The PDF will then be estimated non-parametrically using the Parzen Window and the Parzen Neural Network.\
    The latter will consist of 1 or 2 hidden layers with a sigmoid activation function\
   and the output layer will use either ReLU or sigmoid functions with variable amplitude.
## What is it contained in the report?:
- The report will describe the results and performance of these estimation methods, both graphically and otherwise,\
while varying the cardinality of the extracted point set, the architecture of the neural network, and the hyperparameters of both estimation methods.
