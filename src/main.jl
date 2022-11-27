include("dense_layer.jl")
include("metrics.jl")
include("train.jl")
include("utility_functions.jl")

using MLDatasets:MNIST
using MLBase
x_train, y_train = MNIST.traindata();
x_test, y_test = MNIST.testdata();

m, n = size(x_test[:,:,1])

x_train = reshape(x_train, (m*n,size(x_train)[3]))
x_test = reshape(x_test, (m*n,size(x_test)[3]))

net_layers = [
            Dense("hidden_layer_1", m*n, 256, ReLU!) 
            Dense("hidden_layer_2",256, 128, ReLU!) 
            Dense("output_layer",128, 10, softmax!)
            ];

net_layers, metrics = train(net_layers, x_train, y_train, 100, 256, 0.05);

layers, outputs, ŷ = forward(net_layers, x_test);
confusion_matrix = confusmat(10, y_test.+1, predict(ŷ))
ROC = roc(y_test, predict(ŷ).-1)

model_recall = recall(ROC);
model_precision = precision(ROC)
model_f1_score = f1score(ROC)

print("RECALL: " * string(model_recall) * " | PRECISION: " * string(model_recall) * " | F1: " * string(model_f1_score))
