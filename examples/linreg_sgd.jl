# use SGD for linear regression

using SGDOptim

function risk(pred, θ::Vector{Float64}, X::Matrix{Float64}, y::Vector{Float64})
    u = predict(pred, θ, X)
    0.5 * sumabs2(u - y) / size(X, 2)
end

function linreg_sgd(θ_g::Vector{Float64}, n::Int, σ::Float64)

    # prepare experimental data
    d = length(θ_g) - 1
    X = randn(d, n)
    y = vec(θ_g[1:d]'X) + θ_g[d+1] + σ * randn(n)

    # initialize solution
    θ_0 = zeros(d + 1)

    # optimize
    pred = AffinePredictor()
    θ = sgd(pred, SqrLoss(), θ_0,
        minibatch_seq(X, y, 10),          # configure the way data are supplied
        reg = SqrL2Reg(1.0e-4),           # regularization
        lrate = t->1.0 / (100.0 + t),     # learing rate policy
        cbinterval = 5,                   # how frequently callback is invoked
        callback = gtcompare_trace(θ_g))  # the callback function

    # compare solution with initial guess
    println()
    @printf("Initial:  deviation = %.4e | risk = %.4e\n",
        vecnorm(θ_0 - θ_g), risk(pred, θ_0, X, y))
    @printf("Solution: deviation = %.4e | risk = %.4e\n",
        vecnorm(θ - θ_g), risk(pred, θ, X, y))
end

linreg_sgd([3.0, 5.0, 2.0], 10000, 0.1)
