# use SGD for linear regression

using SGD

function risk(θ::Vector{Float64}, X::Matrix{Float64}, y::Vector{Float64})
    0.5 * sumabs2(X'θ - y) / size(X, 2)
end

function linreg_sgd(θ_g::Vector{Float64}, n::Int, σ::Float64)

    # prepare experimental data
    d = length(θ_g)
    X = randn(d, n)
    y = vec(θ_g'X) + σ * randn(n)

    # initialize solution
    θ_0 = zeros(d)

    # optimize
    θ = sgd(sqrloss!, θ_0,
        minibatch_seq(X, y, 10),
        cbctrl=ByInterval(5),
        callback=gtcompare_trace(θ_g))

    # compare solution with initial guess
    println()
    @printf("Initial:  deviation = %.4e | risk = %.4e\n",
        vecnorm(θ_0 - θ_g), risk(θ_0, X, y))
    @printf("Solution: deviation = %.4e | risk = %.4e\n",
        vecnorm(θ - θ_g), risk(θ, X, y))
end

linreg_sgd([3.0, 5.0], 10000, 0.1)
