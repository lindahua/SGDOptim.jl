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
x = [4.0, 3.0, 2.0]
n = 100
g = zeros(length(θ))

# Squared loss

v = sqrloss!(g, θ, x, 13.0)
@test v == 4.5
@test g == 3.0 * x

v = sqrloss!(g, θ, -x, -12.0)
@test v == 8.0
@test g == -4.0 * (-x)

X = randn(length(θ), n)
y = X'θ + 0.3 * randn(n)

vr, gr = safe_loss_and_grad(sqrloss!, θ, X, y)
v = sqrloss!(g, θ, X, y)
@test_approx_eq v vr
@test_approx_eq g gr
