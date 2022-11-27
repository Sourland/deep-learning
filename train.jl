using BenchmarkTools
using Dates
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
    ∂𝗪[layers[L].name] = (1/m) .* 𝝳[layers[L].name] * outputs[layers[L-1].name]'
    ∂𝗯[layers[L].name] = (1/m) .* sum(𝝳[layers[L].name], dims = 2)

    for l in 2:-1:L-1
        𝝳[layers[l].name] = layers[l+1].weights * 𝝳[layers[l+1].name] .* ∇ReLU!(layers[l].linear_output)
        ∂𝗪[layers[l].name] = (1/m) .* 𝝳[layers[l].name] * outputs[layers[l-1].name]'
        ∂𝗯[layers[l].name] = (1/m) .* sum(𝝳[layers[l].name], dims = 2)
    end

    𝝳[layers[1].name] = layers[2].weights * 𝝳[layers[2].name] .* ∇ReLU!(layers[1].linear_output)
    ∂𝗪[layers[1].name] = (1/m) .* 𝝳[layers[1].name] * input'
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


function one_hot_encoding!( number_of_classes, labels)
    encoded_labels = zeros(number_of_classes, size(labels)[1])
    for idx in 1:size(labels)[1]
        encoded_labels[labels[idx]+1,idx] = 1
    end
    return encoded_labels
end

function train(layers, x_train, y_train, epochs, batch_size)
    y_train_encoded = one_hot_encoding!(maximum(y_train)+1, y_train)
    for e in 1:epochs
        @time begin
        samples = randperm(size(x_train)[2]) 
        for i in 0:convert(Int, size(x_train)[2]/batch_size)-1
            # println("Batch no: " * string(i))
            layers, outputs, current_output = forward(layers, x_train[:,samples[i*batch_size + 1:batch_size+i*batch_size]])
            ∂𝗪, ∂𝗯 = calculate_gradients(layers, x_train[:,samples[i*batch_size + 1:batch_size+i*batch_size]], 
                                        outputs, 
                                        y_train_encoded[:,samples[i*batch_size + 1:batch_size+i*batch_size]], 
                                        current_output)
            layers = update_parameters(layers, ∂𝗪, ∂𝗯, 0.01)
        end
    end
        println("Epoch " * string(e))
    end
    return layers
end

function test(layers, x_test, y_test)
    accuracy = 0
    for i in 1:size(x_test)[2]
        layers, outputs, curr = forward(layers, x_test[:,i])
        accuracy = accuracy + 1*(curr == y_test[i])
        println("Prediction: " * string(argmax(curr)[1]) * " Current: " * string(y_test[i]))
    end
    println("Total accuracy: " * string(accuracy/size(x_test)[2]))

end