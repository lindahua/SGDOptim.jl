

# AbstractLoss subsumes all kinds of loss functions
abstract Loss

abstract ScalarLoss <: Loss
abstract MultinomialLoss <: Loss


## generic implementation of scalar loss

function call(f::ScalarLoss, g::DenseVector, θ::DenseVector, x::DenseVector, y::Real)
    u = dot(θ, x)
    v, dv = value_and_deriv(f, u, y)
    dv == 0.0 ? fill!(g, 0) :
    dv == 1.0 ? copy!(g, x) :
                scale!(g, dv, x)
    return v
end

function call(f::ScalarLoss, g::DenseVector, θ::DenseVector, x::DenseMatrix, y::DenseVector)
    u = x'θ
    v = 0.0
    for i = 1:length(u)
        vi, dvi = value_and_deriv(f, u[i], y[i])
        v += vi
        u[i] = dvi
    end
    A_mul_B!(g, x, u)
    return v
end


## Squared loss (for linear regression)
#
#   loss(θ, x, y) := (1/2) * (θ'x - y)^2
#
type SqrLoss <: ScalarLoss
end

sqrloss! = SqrLoss()

_half(x::Real) = 0.5 * x
_half(x::Float32) = 0.5f0 * x
_half(x::Float64) = 0.5 * x

value_and_deriv(::SqrLoss, u::Real, y::Real) = (r = u - y; v = _half(abs2(r)); (v, r))


## Hinge loss (for SVM)
#
#   loss(θ, x, y) := max(1 - y * θ'x, 0)
#
type HingeLoss <: ScalarLoss
end

hingeloss! = HingeLoss()

function value_and_deriv(::HingeLoss, u::Real, y::Real)
    yu = oftype(u, y) * u
    yu >= one(u) ? (zero(u), zero(u)) : (one(u) - yu, oftype(u, -y))
end


## Logistic loss (for logistic regression)
#
#   loss(θ, x, y) := log(1 + exp(-y * θ'x))
#
type LogisticLoss <: ScalarLoss
end

logisticloss! = LogisticLoss()

function value_and_deriv(::LogisticLoss, u::Real, y::Real)
    yu = oftype(u, y) * u
    if yu >= zero(u)
        e = exp(-yu)
        (log1p(e), -e / (one(e) + e))
    else
        e = exp(yu)
        (log1p(e) - yu, -one(e) / (one(e) + e))
    end
end
