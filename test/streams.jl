
using SGD
using Base.Test


n = 10
k = 8

# SampleSeq

# case 1: X is a matrix, and y is a vector

X = rand(1:100, (5, n))
y = rand(1:100, n)
ord = randperm(n)[1:k]
ss = collect(SampleSeq(X, y, ord))

@test length(ss) == k
for i = 1:k
    @test ss[i] == (X[:,ord[i]], y[ord[i]])
end

# case 2: both X and Y are matrices

X = rand(1:100, (5, n))
Y = rand(1:100, (3, n))
ord = randperm(n)[1:k]
ss = collect(SampleSeq(X, Y, ord))

@test length(ss) == k
for i = 1:k
    @test ss[i] == (X[:, ord[i]], Y[:, ord[i]])
end
