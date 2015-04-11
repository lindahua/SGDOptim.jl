
abstract Predictor

abstract UnivariatePredictor <: Predictor
abstract MultivariatePredictor <: Predictor

## Linear predictor

type LinearPredictor <: UnivariatePredictor
end

nsamples(::LinearPredictor, x::AbstractVector, y::Number) = 1

function nsamples(::LinearPredictor, x::AbstractMatrix, y::AbstractVector)
    n = size(x, 2)
    length(y) == n || throw(DimensionMismatch("Inconsistent number of samples."))
    return n
end

predict(::LinearPredictor, θ::DenseVector, x::DenseVector) = dot(θ, x)
predict(::LinearPredictor, θ::DenseVector, X::DenseMatrix) = X'θ

scaled_grad!(::LinearPredictor, g::DenseVector, c::Real, θ::DenseVector, x::DenseVector) =
    c == zero(c) ? fill!(g, 0) :
    c == one(c) ? copy!(g, x) : scale!(g, c, x)

scaled_grad!(::LinearPredictor, g::DenseVector, c::DenseVector, θ::DenseVector, X::DenseMatrix) =
    A_mul_B!(g, X, c)
