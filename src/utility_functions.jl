
e = MathConstants.e
function relu!(x)
    @. x[x < 0] = 0
end

function softmax!(x)
    exp_values = e .^ x
    x = exp_values / sum(exp_values, dims = 1)
end