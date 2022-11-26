##
using Distributions
using Parameters
using BenchmarkTools
using Dates
include("./utility_functions.jl")
include("./data.jl")
##

##
@with_kw mutable struct Dense
    weights = 0
    bias = 0
    output = 0
    activation = 0
end
##

function Dense(number_of_features::Int, layer_size::Int, activation::Function)
    distribution = Normal()
    weights = 1e-2 * rand(distribution, number_of_features, layer_size)
    bias = zeros(layer_size, 1)
    Dense(weights, bias, undef, x->activation(x))
end

function (d::Dense)(input)
    result = transpose(d.weights) * input .+ d.bias
    d.activation(result)
    return result
end

layer1 = Dense(1024, 512, ReLU!)
layer2 = Dense(512, 256, ReLU!)
layer3 = Dense(256, 10, softmax!)
distribution = Normal()
# input = 1e-2 * rand(distribution, 10, 1)
input = x_test[:,1:256]
@time begin
output1 = layer1(input)
output2 = layer2(output1)
output3 = layer3(output2)
end
##

