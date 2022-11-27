using MLDatasets:MNIST
x_train, y_train = MNIST.traindata();
x_test, y_test = MNIST.testdata();
# x_train = 0.299 * x_train[:,:,1,:] + 0.587* x_train[:,:,2,:] + 0.114 * x_train[:,:,3,:]
# x_test = 0.299 * x_test[:,:,1,:] + 0.587* x_test[:,:,2,:] + 0.114 * x_test[:,:,3,:]
x_train = reshape(x_train, (28*28,size(x_train)[3]))
x_test = reshape(x_test, (28*28,size(x_test)[3]))

