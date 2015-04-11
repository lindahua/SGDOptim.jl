using SGDOptim
using Base.Test


function _predgrad(pred::Predictor, c, θ, x)
    g = similar(θ)
    SGDOptim.scaled_grad!(pred, g, c, θ, x)
    return g
end

## Linear predictor

θ = [3.0, 4.0, 6.0]
x = randn(3)
X = randn(3, 5)
c = rand(size(X, 2))

pred = LinearPredictor()

@test nsamples(pred, θ, x, 0.0) == 1
@test nsamples(pred, θ, X, zeros(5)) == 5

@test_approx_eq dot(θ, x) predict(pred, θ, x)
@test_approx_eq X'θ predict(pred, θ, X)

@test_approx_eq 2.0x _predgrad(pred, 2.0, θ, x)
@test_approx_eq X * c _predgrad(pred, c, θ, X)

## Affine predictor

b = 8.0
a = 2.5
θa = [3.0, 4.0, 6.0, a]

pred = AffinePredictor(b)

@test nsamples(pred, θa, x, 0.0) == 1
@test nsamples(pred, θa, X, zeros(5)) == 5

@test_approx_eq dot(θ, x) + a * b predict(pred, θa, x)
@test_approx_eq X'θ .+ a * b predict(pred, θa, X)

@test_approx_eq [2.0x; 2.0b] _predgrad(pred, 2.0, θa, x)
@test_approx_eq [X * c; b * sum(c)] _predgrad(pred, c, θa, X)

## Multivariate linear predictor

d = 5
k = 3
n = 8

θ = randn(d, k)
x = randn(d)
X = randn(d, n)
c = rand(k)
C = rand(k, n)

pred = MvLinearPredictor()

@test nsamples(pred, θ, x, 0) == 1
@test nsamples(pred, θ, x, zeros(k)) == 1
@test nsamples(pred, θ, X, zeros(Int, n)) == n
@test nsamples(pred, θ, X, zeros(k, n)) == n

@test_approx_eq θ'x predict(pred, θ, x)
@test_approx_eq θ'X predict(pred, θ, X)
@test_approx_eq x * (c') _predgrad(pred, c, θ, x)
@test_approx_eq X * (C') _predgrad(pred, C, θ, X)

# Multivariate affine predictor

b = 3.75
a = randn(k)
θa = [θ; a']
@assert size(θa) == (d+1, k)

pred = MvAffinePredictor(b)

@test nsamples(pred, θa, x, 0) == 1
@test nsamples(pred, θa, x, zeros(k)) == 1
@test nsamples(pred, θa, X, zeros(Int, n)) == n
@test nsamples(pred, θa, X, zeros(k, n)) == n

@test_approx_eq θ'x + a * b predict(pred, θa, x)
@test_approx_eq θ'X .+ a * b predict(pred, θa, X)
@test_approx_eq [x; b] * (c') _predgrad(pred, c, θa, x)
@test_approx_eq [X; b * ones(1,n)] * (C') _predgrad(pred, C, θa, X)
