using Distributions
struct Dense
    number_of_features::Int64
    layer_size::Int64
    weights::Matrix{Float64}
end

function initialize_dense_layer(layer, number_of_features, number_of_neurons)

end