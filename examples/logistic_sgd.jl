# Logistic regression

using SGDOptim

function error_rate(θ::Vector{Float64}, X::Matrix{Float64}, y::Vector{Float64})
    u = X'θ
    countnz(sign(u) .!= sign(y)) / length(y)
end

function logireg_sgd(θ_g::Vector{Float64}, n::Int, σ::Float64)

    # prepare experimental data
    d = length(θ_g)
    X = randn(d, n)
    y = sign(vec(θ_g'X) + σ * randn(n))

    # construct the risk model
    rmodel = riskmodel(LinearPred(d), LogisticLoss())

    # initialize solution
    θ_0 = randn(d)

    # optimize
    θ = sgd(rmodel, θ_0,
        minibatch_seq(X, y, 10),          # configure the way data are supplied
        reg = SqrL2Reg(1.0e-4),           # regularization
        lrate = t->1.0 / (100.0 + t),     # learing rate policy
        cbinterval = 100,                 # how frequently callback is invoked
        callback = simple_trace)          # the callback function

    # compare solution with initial guess
    println()
    @printf("Initial  :  error.rate = %5.2f%%\n", error_rate(θ_0, X, y) * 100.0)
    @printf("Solution :  error.rate = %5.2f%%\n", error_rate(θ,   X, y) * 100.0)
    @printf("gTruth   :  error.rate = %5.2f%%\n", error_rate(θ_g, X, y) * 100.0)
end

logireg_sgd([3.0, 5.0, 2.0], 10000, 0.2)
