using SGDOptim
using Base.Test
using DualNumbers

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

function verify_loss_and_grad(loss::UnivariateLoss, fun, θ::Vector, x::Vector, y::Real)
    u = dot(θ, x)
    fd = fun(dual(u, 1.0), y)
    g = zeros(length(θ))
    v = SGDOptim.loss_and_grad!(linear_predictor, loss, g, θ, x, y)
    @test_approx_eq real(fd) v
    @test_approx_eq epsilon(fd) * x g
end

function verify_loss_and_grads(loss::UnivariateLoss, fun, θ::Vector, X::Matrix, Y::Vector)
    n = size(X, 2)
    U = X'θ
    g = zeros(length(θ))
    v = SGDOptim.loss_and_grad!(linear_predictor, loss, g, θ, X, Y)

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


# data

θ = [1.0, 2.0, 3.0]
x = [0.4, 0.3, 0.2]   # θ'x = 1.6
θx = dot(θ, x)
n = 100
g = zeros(length(θ))
zg = zeros(length(θ))

# Squared loss

_sqrf(u::Dual, y) = 0.5 * abs2(u - y)

verify_value_and_deriv(sqrloss, _sqrf, -3.0:0.25:3.0, [0.0, 1.0, -2.0])

verify_loss_and_grad(sqrloss, _sqrf, θ, x, 1.3)
verify_loss_and_grad(sqrloss, _sqrf, θ, -x, -1.2)

X = randn(length(θ), n)
Y = X'θ + 0.3 * randn(n)
verify_loss_and_grads(sqrloss, _sqrf, θ, X, Y)


# Hinge loss

_hingef(u::Dual, y) = y * real(u) < 1.0 ? 1.0 - y * u : dual(0.0, 0.0)

verify_value_and_deriv(hingeloss, _hingef, -2.0:0.25:2.0, [-1.0, 1.0])

verify_loss_and_grad(hingeloss, _hingef, θ, x, 1)
verify_loss_and_grad(hingeloss, _hingef, θ, 0.5x, 1)
verify_loss_and_grad(hingeloss, _hingef, θ, x, -1)

X = randn(length(θ), n)
Y = sign(X'θ + 0.5 * randn(n))
verify_loss_and_grads(hingeloss, _hingef, θ, X, Y)


# Logistic loss

_logisf(u::Dual, y) = log(1.0 + exp(-y * u))

verify_value_and_deriv(logisticloss, _logisf, -3.0:0.25:3.0, [-1.0, -0.5, 0.5, 1.0])

verify_loss_and_grad(logisticloss, _logisf, θ, x, 1)
verify_loss_and_grad(logisticloss, _logisf, θ, x, -1)
verify_loss_and_grad(logisticloss, _logisf, θ, -x, 0.5)

X = randn(length(θ), n)
Y = 2.0 * rand(n) - 1.0
verify_loss_and_grads(logisticloss, _logisf, θ, X, Y)
