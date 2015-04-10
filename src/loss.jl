

# AbstractLoss subsumes all kinds of loss functions
abstract Loss

abstract ScalarLoss <: Loss
abstract MultinomialLoss <: Loss


# squared loss

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
