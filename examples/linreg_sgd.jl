# use SGD for linear regression

using SGD

function linreg_sgd(theta_g::Vector{Float64}, n::Int, σ::Float64)

    # prepare experimental data
    d = length(theta_g)
    X = randn(d, n)
    y = vec(theta_g'X) + σ * randn(n)

    # initialize solution
    theta_0 = zeros(d)

    # optimize
    sol = sgd(sqrloss!, theta_0,
        SampleSeq(X, y, randperm(n)),
        cbctrl=ByInterval(100),
        callback=gtcompare_trace(theta_g))

    # compare solution with initial guess
    println()
    @printf("Initial:  deviation = %.4e\n", vecnorm(theta_0 - theta_g))
    @printf("Solution: deviation = %.4e\n", vecnorm(sol - theta_g))
end

linreg_sgd([3.0, 5.0], 10000, 0.1)
