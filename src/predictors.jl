
abstract Predictor

abstract UnivariatePredictor <: Predictor
abstract MultivariatePredictor <: Predictor

## Linear predictor

type LinearPredictor <: UnivariatePredictor
end

function nsamples(::LinearPredictor, θ::AbstractVector, x::AbstractVector, y::Number)
    length(x) == length(θ) || throw(DimensionMismatch("Incorrect sample dimensions."))
    return 1
end

function nsamples(::LinearPredictor, θ::AbstractVector, x::AbstractMatrix, y::AbstractVector)
    size(x, 1) == length(θ) || throw(DimensionMismatch("Incorrect sample dimensions."))
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


## Affine predictor

type AffinePredictor{T<:FloatingPoint} <: UnivariatePredictor
    bias::T
end

AffinePredictor{T<:FloatingPoint}(bias::T) = AffinePredictor{T}(bias)
AffinePredictor() = AffinePredictor(1.0)

function nsamples(::AffinePredictor, θ::AbstractVector, x::AbstractVector, y::Number)
    length(x) + 1 == length(θ) || throw(DimensionMismatch("Incorrect sample dimensions."))
    return 1
end

function nsamples(::AffinePredictor, θ::AbstractVector, x::AbstractMatrix, y::AbstractVector)
    size(x, 1) + 1 == length(θ) || throw(DimensionMismatch("Incorrect sample dimensions."))
    n = size(x, 2)
    length(y) == n || throw(DimensionMismatch("Inconsistent number of samples."))
    return n
end

function predict(pred::AffinePredictor, θ::DenseVector, x::DenseVector)
    d = length(x)
    dot(view(θ, 1:d), x) + pred.bias * θ[d+1]
end

function predict(pred::AffinePredictor, θ::DenseVector, X::DenseMatrix)
    d = size(X, 1)
    r = At_mul_B(X, view(θ, 1:d))
    a = pred.bias * θ[d+1]
    @inbounds for i = 1:length(r)
        r[i] += a
    end
    return r
end

function scaled_grad!(pred::AffinePredictor, g::DenseVector, c::Real, θ::DenseVector, x::DenseVector)
    d = length(θ) - 1
    if c == zero(c)
        fill!(g, 0)
    elseif c == one(c)
        copy!(view(g, 1:d), x)
        g[d+1] = pred.bias
    else
        scale!(view(g, 1:d), c, x)
        g[d+1] = pred.bias * c
    end
end

function scaled_grad!(pred::AffinePredictor, g::DenseVector, c::DenseVector, θ::DenseVector, X::DenseMatrix)
    d = length(θ) - 1
    A_mul_B!(view(g, 1:d), X, c)
    g[d+1] = pred.bias * sum(c)
end
