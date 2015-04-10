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
n = 100
g = zeros(length(θ))

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
