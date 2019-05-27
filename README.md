# MNIST-comparison
Comparison of different toy network implementations for MNIST classification (28x28 grayscale images)

* [Linear Classifier](https://github.com/sgttwld/MNIST-comparison/blob/master/1_mnist_LIN.py): Linearly map the `28*28`-simensional input directly with to the 10 outputs (`7,850` parameters, **~92.3% test accuracy**).

* [Simple Feed-Forward Neural Network](https://github.com/sgttwld/MNIST-comparison/blob/master/2_mnist_NN.py): Neural network with one hidden layer of 200 units (`159,010` parameters, **~97.8% test accuracy**).

* [Simple Convolutional Neural Network](https://github.com/sgttwld/MNIST-comparison/blob/master/3_mnist_CNN.py): 
Neural network with one convolutional layer of 32 5x5 filters and one average pooling layer (`46,922` parameters, **~98.6% test accuracy**).

* [Advanced Convolutional Neural Network](https://github.com/sgttwld/MNIST-comparison/blob/master/4_mnist_CNN2.py): Neural network with three convolutional layers (32, 64, and 64 filters of size 3x3), two max pooling layers in-between, and one dense layer with 64 units before the output layer (`93,322` parameters, **~99.1% test accuracy**).
