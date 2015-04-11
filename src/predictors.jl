
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


## Multivariate linear predictor

type MvLinearPredictor <: MultivariatePredictor
end

function nsamples(::MvLinearPredictor, θ::AbstractMatrix, x::AbstractVector, y)
    length(x) == size(θ, 1) || throw(DimensionMismatch("Incorrect sample dimensions."))
    return 1
end

function nsamples(::MvLinearPredictor, θ::AbstractMatrix, x::AbstractMatrix, y::AbstractVecOrMat)
    size(x, 1) == size(θ, 1) || throw(DimensionMismatch("Incorrect sample dimensions."))
    n = size(x, 2)
    size(y, ndims(y)) == n || throw(DimensionMismatch("Inconsistent number of samples."))
    return n
end

predict(::MvLinearPredictor, θ::DenseMatrix, x::DenseVector) = θ'x
predict(::MvLinearPredictor, θ::DenseMatrix, X::DenseMatrix) = θ'X

scaled_grad!(::MvLinearPredictor, g::DenseMatrix, c::DenseVector, θ::DenseMatrix, x::DenseVector) =
    A_mul_Bt!(g, x, c)

scaled_grad!(::MvLinearPredictor, g::DenseMatrix, c::DenseMatrix, θ::DenseMatrix, X::DenseMatrix) =
    A_mul_Bt!(g, X, c)


## Multivariate affine predictor

type MvAffinePredictor{T<:FloatingPoint} <: MultivariatePredictor
    bias::T
end

MvAffinePredictor{T<:FloatingPoint}(bias::T) = MvAffinePredictor{T}(bias)
MvAffinePredictor() = MvAffinePredictor(1.0)

function nsamples(::MvAffinePredictor, θ::AbstractMatrix, x::AbstractVector, y)
    length(x) + 1 == size(θ, 1) || throw(DimensionMismatch("Incorrect sample dimensions."))
    return 1
end

function nsamples(::MvAffinePredictor, θ::AbstractMatrix, x::AbstractMatrix, y::AbstractVecOrMat)
    size(x, 1) + 1 == size(θ, 1) || throw(DimensionMismatch("Incorrect sample dimensions."))
    n = size(x, 2)
    size(y, ndims(y)) == n || throw(DimensionMismatch("Inconsistent number of samples."))
    return n
end

function predict(pred::MvAffinePredictor, θ::DenseMatrix, x::DenseVector)
    d = size(θ, 1) - 1
    k = size(θ, 2)
    r = At_mul_B(view(θ, 1:d, :), x)
    b = pred.bias
    for i = 1:k
        r[i] += θ[d+1, i] * b
    end
    return r
end

function predict(pred::MvAffinePredictor, θ::DenseMatrix, X::DenseMatrix)
    d = size(θ, 1) - 1
    k = size(θ, 2)
    r = At_mul_B(view(θ, 1:d, :), X)
    a = scale!(vec(θ[d+1, :]), pred.bias)
    broadcast!(+, r, r, a)
    return r
end

function scaled_grad!(pred::MvAffinePredictor, g::DenseMatrix, c::DenseVector, θ::DenseMatrix, x::DenseVector)
    d = size(θ, 1) - 1
    A_mul_Bt!(view(g, 1:d, :), x, c)
    b = pred.bias
    for i = 1:size(g, 2)
        g[d+1, i] = c[i] * b
    end
end

function scaled_grad!(pred::MvAffinePredictor, g::DenseMatrix, c::DenseMatrix, θ::DenseMatrix, X::DenseMatrix)
    d = size(θ, 1) - 1
    A_mul_Bt!(view(g, 1:d, :), X, c)
    n = size(X, 2)
    k = size(θ, 2)
    a = scale!(sum(c, 2), pred.bias)
    for i = 1:k
        g[d+1, i] += a[i]
    end
end
