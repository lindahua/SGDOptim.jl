
abstract Regularizer

type NoReg <: Regularizer
end

function value_and_addgrad!{T}(reg::NoReg, g::DenseVecOrMat, θ::DenseVecOrMat{T})
    return zero(T)
end


type SqrL2Reg <: Regularizer
    coef::Float64
end

function value_and_addgrad!{T<:FloatingPoint}(reg::SqrL2Reg, g::DenseVecOrMat, θ::DenseVecOrMat{T})
    axpy!(reg.coef, θ, g)
    convert(T, reg.coef * sumabs2(θ) / 2)
end


type L1Reg <: Regularizer
    coef::Float64
end

function value_and_addgrad!{T<:FloatingPoint}(reg::L1Reg, g::DenseVecOrMat, θ::DenseVecOrMat{T})
    size(g) == size(θ) ||
        throw(DimensionMismatch("The sizes of g and θ are inconsistent."))
    c = convert(T, reg.coef)
    v = zero(T)
    @inbounds for i = 1:length(θ)
        θi = θ[i]
        g[i] += c * sign(θi)
        v += abs(θi)
    end
    return convert(T, c * v)
end


type ElasticReg <: Regularizer
    coef1::Float64
    coef2::Float64
end

function value_and_addgrad!{T<:FloatingPoint}(reg::ElasticReg, g::DenseVecOrMat, θ::DenseVecOrMat{T})
    size(g) == size(θ) ||
        throw(DimensionMismatch("The sizes of g and θ are inconsistent."))
    c1 = convert(T, reg.coef1)
    c2 = convert(T, reg.coef2)
    v1 = zero(T)
    v2 = zero(T)
    @inbounds for i = 1:length(θ)
        θi = θ[i]
        g[i] += (c1 * sign(θi) + c2 * θi)
        v1 += abs(θi)
        v2 += abs2(θi)
    end
    return convert(T, c1 * v1 + c2 * v2 / 2)
end
