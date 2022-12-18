using Distributions
using Parameters
using Octavian
using BenchmarkTools
using Dates

@with_kw mutable struct SVM
    weights = 0
    bias = 0
    learning_rate = 0
    lambda = 0;
end


function SVM(number_of_features::Int, learning_rate, lambda)
    distribution = Normal()
    weights = 1e-2 * rand(distribution, number_of_features,1 )
    bias = 0
    SVM(weights, bias, learning_rate, lambda)
end

function hinge_loss(y, ŷ)
    return maximum([0, 1 - y*ŷ])
end


function fit(svm, X, y, iterations::Int64, kernel = "linear", gamma = 1, c = 1,  d = 2)
    @assert kernel in ["linear" "rbf" "polynomial"]
    x_features, n_samples = size(X)
    for iter in 1:iterations
        for i in 1:n_samples
            
        end
    end
    return svm
end

svm = SVM(m*n, 0.01, 0.1)

fit(svm, x_train, y_train, 50)