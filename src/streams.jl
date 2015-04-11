# Data streams

abstract SampleStream

# auxiliary functions
gets(x::DenseVector, i::Integer) = x[i]
gets(x::DenseVector, i::UnitRange) = view(x, i)
gets(x::DenseVector, i::AbstractVector) = x[i]

gets(x::DenseMatrix, i::Integer) = view(x, :, i)
gets(x::DenseMatrix, i::UnitRange) = view(x, :, i)
gets(x::DenseMatrix, i::AbstractVector) = x[i]


## GenericSampleSeq

type GenericSampleSeq{XS, YS, Ord} <: SampleStream
    xs::XS
    ys::YS
    ord::Ord

    function GenericSampleSeq(xs::XS, ys::YS, ord::Ord)
        nx = size(xs, ndims(xs))
        ny = size(ys, ndims(ys))
        nx == ny ||
            throw(DimensionMismatch("xs and ys must have the same number of samples."))
        new(xs, ys, ord)
    end
end

start(str::GenericSampleSeq) = start(str.ord)
done(str::GenericSampleSeq, s) = done(str.ord, s)

function next(str::GenericSampleSeq, s)
    i, s = next(str.ord, s)
    x = gets(str.xs, i)
    y = gets(str.ys, i)
    return (x, y), s
end


# constructors

function sample_seq(X::DenseVecOrMat, Y::DenseVecOrMat, ord)
    GenericSampleSeq{typeof(X), typeof(Y), typeof(ord)}(X, Y, ord)
end

sample_seq(X::DenseVecOrMat, Y::DenseVecOrMat) = GenericSampleSeq(X, Y, 1:size(X, ndims(X)))


# mini-batch sequences

function batches(n::Int, bsize::Int)
    m = ceil(Int, n / bsize)
    bs = Array(UnitRange{Int}, m)
    last = 0
    for i = 1:m-1
        bs[i] = last+1:last+bsize
        last += bsize
    end
    bs[m] = last+1:n
    return bs
end

minibatch_seq(X::DenseVecOrMat, Y::DenseVecOrMat, bsize::Int) =
    (n = size(X, ndims(X)); sample_seq(X, Y, batches(n, bsize)))

minibatch_seq(X::DenseVecOrMat, Y::DenseVecOrMat, bsize::Int, ord) =
    (n = size(X, ndims(X)); sample_seq(X, Y, batches(n, bsize)[ord]))
