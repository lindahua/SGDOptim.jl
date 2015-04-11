

# AbstractLoss subsumes all kinds of loss functions
abstract Loss

abstract UnivariateLoss <: Loss
abstract MultinomialLoss <: Loss
abstract MultivariateLoss <: Loss


## generic implementation of univariate loss functions

function loss_and_grad!(pred::UnivariatePredictor, loss::UnivariateLoss,
                        g::DenseVector, θ::DenseVector, x::DenseVector, y::Real)

    u = predict(pred, θ, x)
    v, dv = value_and_deriv(loss, u, y)
    scaled_grad!(pred, g, dv, θ, x)
    return v
end

function loss_and_grad!(pred::UnivariatePredictor, loss::UnivariateLoss,
                        g::DenseVector, θ::DenseVector, X::DenseMatrix, Y::DenseVector)

    u = predict(pred, θ, X)
    v = 0.0
    for i = 1:length(u)
        vi, dvi = value_and_deriv(loss, u[i], Y[i])
        v += vi
        u[i] = dvi
    end
    scaled_grad!(pred, g, u, θ, X)
    return v
end


## Squared loss (for linear regression)
#
#   loss(θ, x, y) := (1/2) * (θ'x - y)^2
#
type SqrLoss <: UnivariateLoss
end

value_and_deriv(::SqrLoss, u::Real, y::Real) = (r = u - y; v = half(abs2(r)); (v, r))


## Hinge loss (for SVM)
#
#   loss(θ, x, y) := max(1 - y * θ'x, 0)
#
type HingeLoss <: UnivariateLoss
end

function value_and_deriv(::HingeLoss, u::Real, y::Real)
    yu = oftype(u, y) * u
    yu >= one(u) ? (zero(u), zero(u)) : (one(u) - yu, oftype(u, -y))
end


## Logistic loss (for logistic regression)
#
#   loss(θ, x, y) := log(1 + exp(-y * θ'x))
#
type LogisticLoss <: UnivariateLoss
end

function value_and_deriv(::LogisticLoss, u::Real, y::Real)
    y_ = oftype(u, y)
    yu = y_ * u
    if yu >= zero(u)
        e = exp(-yu)
        (log1p(e), -y_ * e / (one(e) + e))
    else
        e = exp(yu)
        (log1p(e) - yu, -y_ * one(e) / (one(e) + e))
    end
end
