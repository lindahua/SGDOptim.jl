using SGDOptim
using Base.Test


function _predgrad(pred::Predictor, c, θ, x)
    g = similar(θ)
    SGDOptim.scaled_grad!(pred, g, c, θ, x)
    return g
end


θ = [3.0, 4.0, 6.0]
x = randn(3)
X = randn(3, 5)
c = rand(size(X, 2))

## Linear predictor

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
