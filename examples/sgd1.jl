# use SGD for linear regression

using SGD

function linreg_sgd(θg::Vector{Float64}, n::Int, σ::Float64)

    # prepare experimental data
    d = length(θg)
    X = randn(d, n)
    y = vec(θg'X) + σ * randn(n)

    # initialize solution
    θ = zeros(d)

    sgd(sqrloss!, θ, X, y;
        cbctrl=ByInterval(100),
        callback=gtcompare_trace(θg))
end

linreg_sgd([3.0, 5.0], 10000, 0.1)
