# use SGD for linear regression

using SGDOptim

function linreg_sgd(θ_g::Vector{Float64}, n::Int, σ::Float64)

    # prepare experimental data
    d = length(θ_g) - 1
    X = randn(d, n)
    y = vec(θ_g[1:d]'X) + θ_g[d+1] + σ * randn(n)

    # initialize solution
    θ_0 = zeros(d + 1)

    # optimize
    rmodel = riskmodel(AffinePred(d), SqrLoss())

    θ = sgd(rmodel, θ_0,
        minibatch_seq(X, y, 10),          # configure the way data are supplied
        reg = SqrL2Reg(1.0e-4),           # regularization
        lrate = t->1.0 / (100.0 + t),     # learing rate policy
        cbinterval = 5,                   # how frequently callback is invoked
        callback = gtcompare_trace(θ_g))  # the callback function

    # compare solution with initial guess
    println()
    @printf("Initial:  deviation = %.4e | avg.risk = %.4e\n",
        vecnorm(θ_0 - θ_g), value(rmodel, θ_0, X, y) / n)
    @printf("Solution: deviation = %.4e | avg.risk = %.4e\n",
        vecnorm(θ - θ_g), value(rmodel, θ, X, y) / n)
end

linreg_sgd([3.0, 5.0, 2.0], 10000, 0.1)
