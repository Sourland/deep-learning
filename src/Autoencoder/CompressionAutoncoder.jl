using Flux
using Images
using Flux.Data, MLDatasets
using Flux.Data:DataLoader
using Noise

include("train.jl")
device = cpu 

x_train, y_train = CIFAR10.traindata(Float32);
x_test, y_test = CIFAR10.testdata(Float32);


height, width, channels, number_of_pictures = size(x_train)

x_train = reshape(x_train, (m*n,size(x_train)[3]))
x_test = reshape(x_test, (m*n,size(x_test)[3]))
