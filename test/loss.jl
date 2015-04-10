using SGD
using Base.Test


function safe_loss_and_grad(lfun!, θ::Vector, X::Matrix, y::Vector)
    n = size(X, 2)
    v = 0.0
    g = zeros(size(X,1))
    gi = zeros(size(X,1))
    for i = 1:n
        vi = lfun!(gi, θ, X[:,i], y[i])
        v += vi
        g += gi
    end
    return (v, g)
end



θ = [1.0, 2.0, 3.0]
x = [0.4, 0.3, 0.2]   # θ'x = 1.6
θx = dot(θ, x)
n = 100
g = zeros(length(θ))
zg = zeros(length(θ))

# Squared loss

v = sqrloss!(g, θ, x, 1.3)
@test_approx_eq 0.045 v
@test_approx_eq 0.3x g

v = sqrloss!(g, θ, -x, -1.2)
@test_approx_eq 0.08 v
@test_approx_eq 0.4x g

X = randn(length(θ), n)
y = X'θ + 0.3 * randn(n)

vr, gr = safe_loss_and_grad(sqrloss!, θ, X, y)
v = sqrloss!(g, θ, X, y)
@test_approx_eq v vr
@test_approx_eq g gr


# Hinge loss

v = hingeloss!(g, θ, x, 1)
@test_approx_eq 0.0 v
@test_approx_eq zg g

v = hingeloss!(g, θ, 0.5x, 1)
@test_approx_eq 0.2 v
@test_approx_eq -0.5*x g

v = hingeloss!(g, θ, x, -1)
@test_approx_eq 2.6 v
@test_approx_eq x g

X = randn(length(θ), n)
y = sign(X'θ + 0.5 * randn(n))

vr, gr = safe_loss_and_grad(hingeloss!, θ, X, y)
v = hingeloss!(g, θ, X, y)
@test_approx_eq v vr
@test_approx_eq g gr


# Logistic loss

v = logisticloss!(g, θ, x, 1)
vr = log(1.0 + exp(-θx))
gc = - exp(-θx) / (1.0 + exp(-θx))
@test_approx_eq vr v
@test_approx_eq gc * x g

v = logisticloss!(g, θ, -x, 0.5)
vr = log(1.0 + exp(0.5 * θx))
gc = - 0.5 * exp(0.5 * θx) / (1.0 + exp(0.5 * θx))
@test_approx_eq vr v
@test_approx_eq gc * (-x) g

v = logisticloss!(g, θ, x, -1)
vr = log(1.0 + exp(θx))
gc = exp(θx) / (1.0 + exp(θx))
@test_approx_eq vr v
@test_approx_eq gc * x g

X = randn(length(θ), n)
y = 2.0 * rand(n) - 1.0

vr, gr = safe_loss_and_grad(logisticloss!, θ, X, y)
v = logisticloss!(g, θ, X, y)
@test_approx_eq v vr
@test_approx_eq g gr
