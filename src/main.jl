include("dense_layer.jl")
include("metrics.jl")
include("train.jl")
include("utility_functions.jl")

using MLDatasets:MNIST
using EvalMetrics
x_train, y_train = MNIST.traindata();
x_test, y_test = MNIST.testdata();

m, n = size(x_test[:,:,1])

x_train = reshape(x_train, (m*n,size(x_train)[3]))
x_test = reshape(x_test, (m*n,size(x_test)[3]))

net_layers = [
            Dense("hidden_layer_1", m*n, 256, ReLU!) 
            Dense("hidden_layer_2",256, 128, ReLU!) 
            Dense("output_layer",128, 10, softmax!)
            ]

net_layers, metrics = train(net_layers, x_train, y_train, 10, 256, 0.05);

layers, outputs, ŷ = forward(net_layers, x_test)
confusion_matrix = ConfusionMatrix(y_test, predict(ŷ).-1)
