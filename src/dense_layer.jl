##
using Distributions
using Parameters
include("./utility_functions.jl")
include("./data.jl")
##

##
@with_kw mutable struct Dense
    name = 0
    weights = 0
    bias = 0
    linear_output = 0
    activation = 0
end
##

function Dense(name ,number_of_features::Int, layer_size::Int, activation::Function)
    distribution = Normal()
    weights = 1e-2 * rand(distribution, number_of_features, layer_size)
    bias = zeros(layer_size, 1)
    Dense(name, weights, bias, undef, x->activation(x))
end

function (layer::Dense)(input)
    result = transpose(layer.weights) * input .+ layer.bias
    layer.linear_output = copy(result)
    layer.activation(result)
    return result
end

layer1 = Dense("l1", 32*32, 512, ReLU!)
layer2 = Dense("l2",512, 256, ReLU!)
layer3 = Dense("l3",256, 10, softmax!)