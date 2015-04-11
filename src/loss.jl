

# AbstractLoss subsumes all kinds of loss functions
abstract Loss

abstract UnivariateLoss <: Loss
abstract MultivariateLoss <: Loss


## generic implementation of univariate loss functions

function value_and_grad!(pred::UnivariatePredictor, loss::UnivariateLoss,
                        g::DenseVector, θ::DenseVector, x::DenseVector, y::Real)

    u = predict(pred, θ, x)
    v, dv = value_and_deriv(loss, u, y)
    scaled_grad!(pred, g, dv, θ, x)
    return v
end

function value_and_grad!(pred::UnivariatePredictor, loss::UnivariateLoss,
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


## generic implementation of multinomial loss functions

function value_and_grad!(pred::MultivariatePredictor, loss::MultivariateLoss,
                         g::DenseMatrix, θ::DenseMatrix, x::DenseVector, y)

    u = predict(pred, θ, x)
    v = value_and_deriv!(loss, u, y)  # derivatives written to u
    scaled_grad!(pred, g, u, θ, x)
    return v
end

function value_and_grad!(pred::MultivariatePredictor, loss::MultivariateLoss,
                         g::DenseMatrix, θ::DenseMatrix, X::DenseMatrix, y)
    u = predict(pred, θ, X)
    v = 0.0
    for i = 1:size(X,2)
        vi = value_and_deriv!(loss, view(u, :, i), gets(y, i))
        v += vi
    end
    scaled_grad!(pred, g, u, θ, X)
    return v
end


## Multinomial logistic loss (for Multinomial logistic regression)

type MultiLogisticLoss <: MultivariateLoss
end

function value_and_deriv!{T<:FloatingPoint}(::MultiLogisticLoss, u::DenseVector{T}, y::Integer)
    umax = maximum(u)
    uy = u[y]

    k = length(u)
    s = zero(T)
    @inbounds for i = 1:k
        ui = exp(u[i] - umax)
        u[i] = ui
        s += ui
    end

    @inbounds for i = 1:k
        u[i] /= s
    end
    u[y] -= one(T)

    return umax - uy + log(s)
end
