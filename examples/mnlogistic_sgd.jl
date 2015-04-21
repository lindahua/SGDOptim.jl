# Multinomial Logistic regression

using SGDOptim
using ArrayViews

function error_rate(W::Matrix{Float64}, X::Matrix{Float64}, y::Vector{Int})
    u = W * X
    r = Int[indmax(view(u,:,i)) for i = 1:size(X,2)]
    countnz(r .!= y) / length(y)
end

function mnlogireg_sgd(Wg::Matrix{Float64}, n::Int, σ::Float64)

    # prepare experimental data
    k, d = size(Wg)
    X = randn(d, n)
    u = Wg * X + σ * randn(k, n)
    y = Int[indmax(view(u,:,i)) for i = 1:n]

    # construct the risk model
    rmodel = riskmodel(MvLinearPred(d, k), MultiLogisticLoss())

    # initialize solution
    W0 = randn(k, d)

    # optimize
    W = sgd(rmodel, W0,
        minibatch_seq(X, y, 10),          # configure the way data are supplied
        reg = SqrL2Reg(1.0e-4),           # regularization
        lrate = t->1.0 / (100.0 + t),     # learing rate policy
        cbinterval = 100,                 # how frequently callback is invoked
        callback = simple_trace)          # the callback function

    # compare solution with initial guess
    println()
    @printf("Initial  :  error.rate = %5.2f%%\n", error_rate(W0, X, y) * 100.0)
    @printf("Solution :  error.rate = %5.2f%%\n", error_rate(W,  X, y) * 100.0)
    @printf("gTruth   :  error.rate = %5.2f%%\n", error_rate(Wg, X, y) * 100.0)
end

mnlogireg_sgd(randn(3, 5), 10000, 0.2)
