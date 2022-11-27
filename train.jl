using BenchmarkTools
using Dates
using Random
using Octavian
include("src/dense_layer.jl")


function forward(layers, input)
    L = maximum(size(layers))
    outputs = Dict()
    current_output = layers[1](input)
    outputs[layers[1].name] = current_output
    for l in 2:L
        current_output = layers[l](current_output)
        outputs[layers[l].name] = current_output
    end    
    return layers, outputs, current_output
end

function calculate_gradients(layers, input, outputs, Y, Ŷ)
    L = maximum(size(layers))
    m = size(input)[2]
    ∂𝗪 = Dict()
    ∂𝗯 = Dict()
    𝝳 = Dict()
    
    𝝳[layers[L].name] = Ŷ - Y 
    ∂𝗪[layers[L].name] = (1/m) .* matmul(𝝳[layers[L].name],outputs[layers[L-1].name]')
    ∂𝗯[layers[L].name] = (1/m) .* sum(𝝳[layers[L].name], dims = 2)

    for l in 2:-1:L-1
        𝝳[layers[l].name] = matmul(layers[l+1].weights, 𝝳[layers[l+1].name]) .* ∇ReLU!(layers[l].linear_output)
        ∂𝗪[layers[l].name] = (1/m) .* matmul(𝝳[layers[l].name], outputs[layers[l-1].name]')
        ∂𝗯[layers[l].name] = (1/m) .* sum(𝝳[layers[l].name], dims = 2)
    end

    𝝳[layers[1].name] = matmul(layers[2].weights, 𝝳[layers[2].name]) .*∇ReLU!(layers[1].linear_output)
    ∂𝗪[layers[1].name] = (1/m) .* matmul(𝝳[layers[1].name], Float64.(input'))
    ∂𝗯[layers[1].name] = (1/m) .* sum(𝝳[layers[1].name], dims = 2)
    
    return  ∂𝗪, ∂𝗯 
end


function update_parameters(layers, ∂𝗪, ∂𝗯, 𝞰)
    L = maximum(size(layers))
    for l in 1:L
        layers[l].weights = layers[l].weights - 𝞰 .* ∂𝗪[layers[l].name]'
        layers[l].bias = layers[l].bias - 𝞰 .* ∂𝗯[layers[l].name]
    end
    return layers
end


function predict(y)
    return map(argmax, eachcol(y))
end

layers = [layer1 layer2 layer3]


function one_hot_encoding( number_of_classes, labels)
    encoded_labels = zeros(number_of_classes, size(labels)[1])
    for idx in 1:size(labels)[1]
        encoded_labels[labels[idx]+1,idx] = 1
    end
    return encoded_labels
end

function train(layers, x_train, y_train, epochs, batch_size)
    evaluation_samples = randperm(size(x_train)[2])
    evaluation_idx = evaluation_samples[1:convert(Int, floor(0.4*size(x_train)[2]))]
    x_val = x_train[:,evaluation_idx]
    y_val = y_train[evaluation_idx]

    x_train = x_train[:,setdiff(1:end, Tuple(evaluation_idx))]
    y_train = y_train[setdiff(1:end, Tuple(evaluation_idx))]

    y_train_encoded = one_hot_encoding(maximum(y_train)+1, y_train)
    for e in 1:epochs
        @time begin
        samples = randperm(size(x_train)[2]) 
        for i in 0:convert(Int, floor(size(x_train)[2]/batch_size))-1
            if (i*batch_size + batch_size+ 1) < size(x_train)[2]
                currext_idx = samples[i*batch_size + 1:i*batch_size + batch_size]
            else
                currext_idx = samples[i*batch_size + 1:end]
            end
            layers, outputs, current_output = forward(layers, x_train[:,currext_idx])
            ∂𝗪, ∂𝗯 = calculate_gradients(layers, x_train[:,currext_idx], 
                                        outputs, 
                                        y_train_encoded[:,currext_idx], 
                                        current_output)
            layers = update_parameters(layers, ∂𝗪, ∂𝗯, 0.01)
            # val_accuracy = test(layers, x_val, y_val)
        end
        println("Epoch " * string(e))

    end
        
    end
    return layers
end

function test(layers, x_test, y_test)
    y_test = one_hot_encoding(maximum(y_test)+1, y_test)
    accuracy = 0
    for i in 1:size(x_test)[2]
        layers, outputs, curr = forward(layers, x_test[:,i])
        accuracy = accuracy + 1*(argmax(curr)[1]  == argmax(y_test[:,i]))
        # println("Prediction: " * string(argmax(curr)[1]-1) * " Current: " * string(argmax(y_test[:,i])[1]-1))
    end
    return accuracy./size(y_test)[2]
end