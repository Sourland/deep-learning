
e = MathConstants.e
function relu!(x)
    @. x[x < 0] = 0
end

function softmax!(x)
    exp_values = e .^ x
    x = exp_values / sum(exp_values, dims = 1)
end

function categorical_cross_entropy(y_pred, y_true)
    confidence = sum(y_pred * y_true, dims=1)
    return -log.(confidence)
end