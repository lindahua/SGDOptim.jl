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

pred = LinearPredictor()

@test_approx_eq dot(θ, x) predict(pred, θ, x)
@test_approx_eq X'θ predict(pred, θ, X)

@test_approx_eq 2.0x _predgrad(pred, 2.0, θ, x)
@test_approx_eq X * c _predgrad(pred, c, θ, X)
