using MLDatasets:CIFAR10
using Images, ImageInTerminal
x_train, y_train = CIFAR10.traindata();
x_test, y_test = CIFAR10.testdata();
x_train = 0.299 * x_train[:,:,1,:] + 0.587* x_train[:,:,2,:] + 0.114 * x_train[:,:,3,:]
x_test = 0.299 * x_test[:,:,1,:] + 0.587* x_test[:,:,2,:] + 0.114 * x_test[:,:,3,:]
x_train = reshape(x_train, (32*32,size(x_train)[3]))
x_test = reshape(x_test, (32*32,size(x_test)[3]))



