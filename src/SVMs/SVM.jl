##
using Distributions
using Parameters
using Octavian
using BenchmarkTools
using Dates
using JuMP
using Ipopt
using SCS
include("kernels.jl")
include("../NNs/main.jl")
##
@with_kw mutable struct SVM
    weights = 0
    bias = 0
end


function SVM(number_of_features::Int)
    distribution = Normal()
    weights = zeros(number_of_features)
    bias = 0
    SVM(weights, bias)
end

function fit(svm, X, y, kernel = "linear", mode = "hard", C = 10)
    X_pos = X_[:,y.==1]
    X_neg = X_[:,y.==1]
    n_features, n_samples = size(X)
    optimizer = Model(Ipopt.Optimizer)
    @variable(optimizer, alpha[1:n_samples])
    println("SETTING LAGRANGE MULTIPLIER CONSTRAINTS")
    for i = 1:n_samples
        @constraint(optimizer, alpha[i] >= 0)
    end
    println("CHECKING FOR SOFT MARGINS AND ADDING THE CONSTRAINS RESPECTIVELY")
    if mode == "soft"
        for i = 1:n_samples
            @constraint(optimizer, alpha[i] <= C)
        end
    end
    println("ADDING KARUSH-KUHN-TUCKER CONDITIONS")
    @constraint(optimizer, alpha' * y == 0)

    println("SET OBJECTIVE")
    if kernel == "linear"
        @objective(optimizer, Max, sum(alpha) - sum(alpha[i]*alpha[j]*y[i]*y[j]*linear_kernel(X[:,i], X[:,j]) for i = 1:n_samples, j = 1:n_samples))
    elseif kernel == "rbf"
        @objective(optimizer, Max, sum(alpha) - sum(alpha[i]*alpha[j]*y[i]*y[j]*rbf_kernel(X[:,i], X[:,j]) for i = 1:n_samples, j = 1:n_samples))
    else
        @objective(optimizer, Max, sum(alpha) - sum(alpha[i]*alpha[j]*y[i]*y[j]*polynomial_kernel(X[:,i], X[:,j]) for i = 1:n_samples, j = 1:n_samples))
    end
    println("BEGIN OPTIMIZATION PROBLEM SOLVER")
    optimize!(optimizer)
    a = value.(alpha)
    print(a)
    for i = 1:n_samples
        # print()
        svm.weights = svm.weights .+ a[i]*y[i].*rbf_kernel(X[:,i], zeros(n_features))
        svm.bias = mean(y .- svm.weights'rbf_kernel(X, zeros(n_features, n_samples)))
    end
    return svm
end


function predict(svm, X)
    sign.(svm.weights'X .+ svm.bias)
end


function svm(pos_data, neg_data, solver=() -> SCS.Optimizer(verbose=0))
    # Create variables for the separating hyperplane w'*x = b.
    w = Variable(n)
    b = Variable()
    # Form the objective.
    obj = sumsquares(w) + C*sum(max(1+b-w'*pos_data, 0)) + C*sum(max(1-b+w'*neg_data, 0))
    # Form and solve problem.
    problem = minimize(obj)
    solve!(problem, solver)
    return evaluate(w), evaluate(b)
end;