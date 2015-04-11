# for profiling and optimizing the implementation of SGD

using SGDOptim
using Base.Profile

function risk(pred, θ::Vector{Float64}, X::Matrix{Float64}, y::Vector{Float64})
    u = predict(pred, θ, X)
    0.5 * sumabs2(u - y) / size(X, 2)
end

function linreg_sgd(θ_g::Vector{Float64}, n::Int, σ::Float64)

    # prepare experimental data
    d = length(θ_g)
    X = randn(d, n)
    y = vec(θ_g'X) + σ * randn(n)

    # initialize solution
    θ_0 = zeros(d)

    # optimize
    pred = LinearPredictor()
    loss = SqrLoss()
    stream = minibatch_seq(X, y, 10)
    reg = SqrL2Reg(1.0e-4)
    g = zeros(d)
    lrate = SGDOptim.default_lrate

    init = copy(θ_0)
    sgd!(pred, loss, reg, init, stream, lrate, 0, simple_trace)

    @profile sgd!(pred, loss, reg, init, stream, lrate, 0, simple_trace)
    Profile.print()

    @time for i = 1:10
        sgd!(pred, loss, reg, init, stream, lrate, 0, simple_trace)
    end
end

linreg_sgd([3.0, 5.0, 2.0], 100000, 0.1)
