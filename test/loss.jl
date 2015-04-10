using SGD
using Base.Test

θ = [1.0, 2.0, 3.0]
x = [4.0, 3.0, 2.0]
g = zeros(3)

v = sqrloss!(g, θ, x, 13.0)
@test v == 4.5
@test g == 3.0 * x
