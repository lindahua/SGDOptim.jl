
using SGDOptim
using Base.Test

n = 10
k = 8

# SampleSeq

# case 1: X is a matrix, and y is a vector

X = rand(1:100, (5, n))
y = rand(1:100, n)
ord = randperm(n)[1:k]
ss = collect(sample_seq(X, y, ord))

@test length(ss) == k
for i = 1:k
    j = ord[i]
    @test ss[i] == (X[:,j], y[j])
end

# case 2: both X and Y are matrices

X = rand(1:100, (5, n))
Y = rand(1:100, (3, n))
ord = randperm(n)[1:k]
ss = collect(sample_seq(X, Y, ord))

@test length(ss) == k
for i = 1:k
    j = ord[i]
    @test ss[i] == (X[:, j], Y[:, j])
end

# case 3: mini-batches in natural order

@test SGDOptim.batches(10, 3) == UnitRange{Int}[1:3, 4:6, 7:9, 10:10]
@test SGDOptim.batches(12, 3) == UnitRange{Int}[1:3, 4:6, 7:9, 10:12]

X = rand(1:100, (5, n))
y = rand(1:100, n)

bs = UnitRange{Int}[1:3, 4:6, 7:9, 10:10]
ss = collect(minibatch_seq(X, y, 3))

@test length(ss) == length(bs)
for i = 1:length(bs)
    b = bs[i]
    @test ss[i] == (X[:, b], y[b])
end

ord = [4, 2, 3, 1, 2]
ss = collect(minibatch_seq(X, y, 3, ord))

@test length(ss) == length(ord)
for i = 1:length(ord)
    b = bs[ord[i]]
    @test ss[i] == (X[:, b], y[b])
end
