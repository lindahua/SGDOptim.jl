# predictors
abstract Predictor

type LinearPredictor <: Predictor
    dim::Int
end

paramdim(p::LinearPredictor) = p.dim
sampledim(p::LinearPredictor) = p.dim

function call(p::LinearPredictor, θ::DenseVector, x::DenseVector)
    length(θ) == length(x) == p.dim ||
        throw(DimensionMismatch("Incorrect input dimensions."))
    dot(θ, x)
end
