

# AbstractLoss subsumes all kinds of loss functions
abstract Loss

abstract ScalarLoss <: Loss
abstract MultinomialLoss <: Loss


## Squared loss (for linear regression)
#
#   loss(θ, x, y) := (1/2) * (θ'x - y)^2
#
type SqrLoss <: ScalarLoss
end

sqrloss! = SqrLoss()

# for a single sample
function call(::SqrLoss, g::DenseVector, θ::DenseVector, x::DenseVector, y::Real)
    r = dot(θ, x) - y
    scale!(g, r, x)
    0.5 * abs2(r)
end

# for a sample batch
function call(::SqrLoss, g::DenseVector, θ::DenseVector, x::DenseMatrix, y::DenseVector)
    r = x'θ - y
    A_mul_B!(g, x, r)
    return 0.5 * sumabs2(r)
end


## Hinge loss (for SVM)
#
#   loss(θ, x, y) := max(1 - y * θ'x, 0)
#
type HingeLoss <: ScalarLoss
end

hingeloss! = HingeLoss()

# for a single sample
function call(::HingeLoss, g::DenseVector, θ::DenseVector, x::DenseVector, y::Real)
    r = dot(θ, x) * y
    if r >= one(r)
        fill!(g, 0)
    else
        scale!(g, -y, x)
    end
    max(1.0 - r, 0.0)
end

# for a sample batch
function call(::HingeLoss, g::DenseVector, θ::DenseVector, x::DenseMatrix, y::DenseVector)
    u = x'θ
    fill!(g, 0)
    v = zero(u)
    for i = 1:length(u)
        uy = u[i] * y[i]
        if uy < one(uy)
            axpy!(-y[i], x, g)
            v += (one(uy) - uy)
        end
    end
    return v
end


## Logistic loss (for logistic regression)
#
#   loss(θ, x, y) := log(1 + exp(-y * θ'x))
#
type LogisticLoss <: ScalarLoss
end

logisticloss! = LogisticLoss()

# for a single sample

# computes (log(1 + exp(-x)), exp(-x) / (1 + exp(-x)))
# in a numerically stable way
#
function _logistic_deriv(x::Real)
    if x >= zero(x)
        e = exp(-x)
        (log1p(e), e / (one(e) + e))
    else
        e = exp(x)
        (log1p(e) - x, one(e) / (one(e) + e))
    end
end

function call(::LogisticLoss, g::DenseVector, θ::DenseVector, x::DenseVector, y::Real)
    r = dot(θ, x) * y
    v, dv = _logistic_deriv(r)
    scale!(g, -y * dv, x)
    return v
end

function call(::LogisticLoss, g::DenseVector, θ::DenseVector, x::DenseMatrix, y::DenseVector)
    u = x'θ
    v = 0.0
    for i = 1:length(u)
        vi, u[i] = _logistic_deriv(y[i] * u[i])
        v += vi
        u[i] *= (-y[i])
    end
    A_mul_B!(g, X, u)
    return v
end
