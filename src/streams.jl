# Data streams

abstract SampleStream

# auxiliary functions
gets(x::DenseVector, i::Integer) = x[i]
gets(x::DenseVector, i::UnitRange) = view(x, i)
gets(x::DenseVector, i::AbstractVector) = x[i]

gets(x::DenseMatrix, i::Integer) = view(x, :, i)
gets(x::DenseMatrix, i::UnitRange) = view(x, :, i)
gets(x::DenseMatrix, i::AbstractVector) = x[i]


type SampleSeq{XS, YS, Ord} <: SampleStream
    xs::XS
    ys::YS
    ord::Ord

    function SampleSeq(xs::XS, ys::YS, ord::Ord)
        nx = size(xs, ndims(xs))
        ny = size(ys, ndims(ys))
        nx == ny ||
            throw(DimensionMismatch("xs and ys must have the same number of samples."))
        new(xs, ys, ord)
    end
end

start(str::SampleSeq) = start(str.ord)
done(str::SampleSeq, s) = done(str.ord, s)

function next(str::SampleSeq, s)
    i, s = next(str.ord, s)
    x = gets(str.xs, i)
    y = gets(str.ys, i)
    return (x, y), s
end

function SampleSeq(X::DenseVecOrMat, Y::DenseVecOrMat, ord)
    SampleSeq{typeof(X), typeof(Y), typeof(ord)}(X, Y, ord)
end

SampleSeq(X::DenseVecOrMat, Y::DenseVecOrMat) = SampleSeq(X, Y, 1:size(X, ndims(X)))
