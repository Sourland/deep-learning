using Distributions
using Parameters


@with_kw mutable struct Dense
    weights = 0
    bias = 0
    output = 0
end

function Dense(number_of_features::Int, layer_size::Int)
    distribution = Normal()
    weights = 1e-2 * rand(distribution, number_of_features, layer_size)
    bias = zeros(layer_size, 1)
    Dense(weights, bias, undef)
end

function relu!(input)
    negative_elements_idx = Tuple.(findall(x->x<0, input))
    """TODO: Make this vectorized"""
    for idx in negative_elements_idx
        input[idx[1]] = 0
    end
end

function forward!(layer, inputs)
    layer.output = relu!(layer.weights * inputs + layer.bias)
end



layer = Dense(10, 10)
distribution = Normal()
input = 1e-2 * rand(distribution, 10, 1)
forward!(layer, input)
layer.output


