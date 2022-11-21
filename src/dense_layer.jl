##
using Distributions
using Parameters
include("./utility_functions.jl")
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

#
function forward(layer, inputs)
    result = transpose(layer.weights) * inputs + layer.bias
    layer.activation(result)
    print("raccoooooooon")
    return result
end
##

##
layer1 = Dense(10, 20, relu!)
layer2 = Dense(20, 10, softmax!)
distribution = Normal()
input = 1e-2 * rand(distribution, 10, 1)
layer1.output = forward(layer1, input)
layer2.output = forward(layer2, layer1.output)
##

