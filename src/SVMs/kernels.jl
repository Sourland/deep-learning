using LinearAlgebra
function polynomial_kernel(x, x_prime, constant = 1, d = 4)
    return (x' * x_prime .+ constant) .^ d
end

function rbf_kernel(x, x_prime, gamma = 1)
    return exp.(-gamma * norm.(eachcol(x-x_prime)) .^ 2)[1]
end

function linear_kernel(x, x_prime)
    return x' * x_prime
end