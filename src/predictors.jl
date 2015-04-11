
abstract Predictor

abstract UnivariatePredictor <: Predictor
abstract MultivariatePredictor <: Predictor

type LinearPredictor <: UnivariatePredictor
end

nsamples(::LinearPredictor, x::DenseVecOrMat) = size(x, 2)

predict(::LinearPredictor, θ::DenseVector, x::DenseVector) = dot(θ, x)
predict(::LinearPredictor, θ::DenseVector, X::DenseMatrix) = X'θ

scaled_grad!(::LinearPredictor, g::DenseVector, c::Real, θ::DenseVector, x::DenseVector) =
    c == zero(c) ? fill!(g, 0) :
    c == one(c) ? copy!(g, x) : scale!(g, c, x)

scaled_grad!(::LinearPredictor, g::DenseVector, c::DenseVector, θ::DenseVector, X::DenseMatrix) =
    A_mul_B!(g, X, c)
