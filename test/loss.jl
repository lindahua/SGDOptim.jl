using SGDOptim
using Base.Test
using DualNumbers


## tools for checking function value and gradients

function verify_value_and_deriv(loss::UnivariateLoss, fun, us::AbstractVector{Float64}, ys::AbstractVector{Float64})
    for y in ys
        for u in us
            v, dv = SGDOptim.value_and_deriv(loss, u, y)
            fd = fun(dual(u, 1.0), y)
            @test_approx_eq real(fd) v
            @test_approx_eq epsilon(fd) dv
        end
    end
end

function _vds(fun, u::Vector{Float64}, y)  # compute value and derivatives using dual numbers
    k = length(u)
    dv = zeros(k)
    v = real(fun(dual(u), y))
    for i = 1:k
        ep = zeros(k)
        ep[i] = 1.0
        dv[i] = epsilon(fun(dual(u, ep), y))
    end
    return v, dv
end

function verify_value_and_deriv(loss::MultivariateLoss, fun, u::Vector{Float64}, y)
    k = length(u)
    dvs = copy(u)
    v = SGDOptim.value_and_deriv!(loss, dvs, y)
    rv, rdvs = _vds(fun, u, y)
    @test_approx_eq rv v
    @test_approx_eq rdvs dvs
end

function verify_loss_and_grad(loss::UnivariateLoss, fun, θ::Vector, x::Vector, y::Real)
    u = dot(θ, x)
    fd = fun(dual(u, 1.0), y)
    g = zeros(length(θ))
    v = SGDOptim.value_and_grad!(LinearPredictor(), loss, g, θ, x, y)
    @test_approx_eq real(fd) v
    @test_approx_eq epsilon(fd) * x g
end

function verify_loss_and_grads(loss::UnivariateLoss, fun, θ::Vector, X::Matrix, Y::Vector)
    n = size(X, 2)
    U = X'θ
    g = zeros(length(θ))
    v = SGDOptim.value_and_grad!(LinearPredictor(), loss, g, θ, X, Y)

    rv = 0.0
    rg = zeros(length(θ))
    for i = 1:n
        u = U[i]
        y = Y[i]
        fd = fun(dual(u, 1.0), y)
        rv += real(fd)
        rg += epsilon(fd) * X[:,i]
    end

    @test_approx_eq rv v
    @test_approx_eq rg g
end

function verify_loss_and_grad(loss::MultivariateLoss, fun, θ::Matrix, x::Vector, y)
    u = θ'x
    rv, rdvs = _vds(fun, u, y)
    rg = A_mul_Bt(x, rdvs)
    g = zeros(size(θ))
    v = SGDOptim.value_and_grad!(MvLinearPredictor(), loss, g, θ, x, y)

    @test_approx_eq rv v
    @test_approx_eq rg g
end

function verify_loss_and_grad(loss::MultivariateLoss, fun, θ::Matrix, X::Matrix, Y)
    U = θ'X

    rv = 0.0
    rg = zeros(size(θ))
    for i = 1:size(X,2)
        y_i = SGDOptim.gets(Y, i)
        rv_i, rdvs_i = _vds(fun, U[:,i], y_i)
        rg_i = A_mul_Bt(X[:,i], rdvs_i)
        rv += rv_i
        rg += rg_i
    end
    g = zeros(size(θ))
    v = SGDOptim.value_and_grad!(MvLinearPredictor(), loss, g, θ, X, Y)

    @test_approx_eq rv v
    @test_approx_eq rg g
end



# data

θ = [1.0, 2.0, 3.0]
x = [0.4, 0.3, 0.2]   # θ'x = 1.6
θx = dot(θ, x)
n = 100
g = zeros(length(θ))
zg = zeros(length(θ))

# Squared loss

_sqrf(u::Dual, y) = 0.5 * abs2(u - y)

verify_value_and_deriv(SqrLoss(), _sqrf, -3.0:0.25:3.0, [0.0, 1.0, -2.0])

verify_loss_and_grad(SqrLoss(), _sqrf, θ, x, 1.3)
verify_loss_and_grad(SqrLoss(), _sqrf, θ, -x, -1.2)

X = randn(length(θ), n)
Y = X'θ + 0.3 * randn(n)
verify_loss_and_grads(SqrLoss(), _sqrf, θ, X, Y)


# Hinge loss

_hingef(u::Dual, y) = y * real(u) < 1.0 ? 1.0 - y * u : dual(0.0, 0.0)

verify_value_and_deriv(HingeLoss(), _hingef, -2.0:0.25:2.0, [-1.0, 1.0])

verify_loss_and_grad(HingeLoss(), _hingef, θ, x, 1)
verify_loss_and_grad(HingeLoss(), _hingef, θ, 0.5x, 1)
verify_loss_and_grad(HingeLoss(), _hingef, θ, x, -1)

X = randn(length(θ), n)
Y = sign(X'θ + 0.5 * randn(n))
verify_loss_and_grads(HingeLoss(), _hingef, θ, X, Y)


# Logistic loss

_logisf(u::Dual, y) = log(1.0 + exp(-y * u))

verify_value_and_deriv(LogisticLoss(), _logisf, -3.0:0.25:3.0, [-1.0, -0.5, 0.5, 1.0])

verify_loss_and_grad(LogisticLoss(), _logisf, θ, x, 1)
verify_loss_and_grad(LogisticLoss(), _logisf, θ, x, -1)
verify_loss_and_grad(LogisticLoss(), _logisf, θ, -x, 0.5)

X = randn(length(θ), n)
Y = 2.0 * rand(n) - 1.0
verify_loss_and_grads(LogisticLoss(), _logisf, θ, X, Y)


# Multinomial Logistic loss

d = 5
k = 3
n = 8
θ = randn(d, k)
X = randn(d, n)
Y = [1,2,3,2,3,1,3,2]
@assert length(Y) == n

_mlogisf(u::Vector{Dual{Float64}}, y::Int) = log(sum(exp(u))) - u[y]

for y = 1:4
    verify_value_and_deriv(MultiLogisticLoss(), _mlogisf, [-2.0, 0.0, 1.0, 3.0], y)
end

for i = 1:n
    x = copy(X[:,i])
    for y = 1:k
        verify_loss_and_grad(MultiLogisticLoss(), _mlogisf, θ, x, y)
    end
end

verify_loss_and_grad(MultiLogisticLoss(), _mlogisf, θ, X, Y)
