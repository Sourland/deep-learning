
e = MathConstants.e
function ReLU!(x)
    @. x[x < 0] = 0
end

function ∇ReLU!(x)
    return x.>0disc
end

function softmax!(x)
    exp_values = e .^ x
    output = exp_values ./ sum(exp_values, dims = 1)
    x[:] = output
end

function categorical_cross_entropy(ŷ, y)
    product = ŷ .* y
    confidence = sum(product, dims = 1)
    return -log.(confidence)
end