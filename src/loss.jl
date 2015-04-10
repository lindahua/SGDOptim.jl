

# AbstractLoss subsumes all kinds of loss functions
abstract Loss

abstract ScalarLoss <: Loss
abstract MultinomialLoss <: Loss


# squared loss

type SqrLoss end

function call(::SqrLoss, g::DenseVector, θ::DenseVector, x::DenseVector, y::Real)
    r = dot(θ, x) - y
    scale!(g, r, x)
    0.5 * abs2(r)
end

sqrloss! = SqrLoss()
