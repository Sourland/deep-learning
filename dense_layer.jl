using Distributions
using Parameters


@with_kw struct Dense
    weights = 0
    bias = 0
end

function Dense(number_of_features::Int, layer_size::Int)
    distribution = Normal()
    weights = 1e-2 * rand(distribution, number_of_features, layer_size)
    bias = zeros(layer_size, 1)
    Dense(weights, bias)
end

Dense(10, 10)