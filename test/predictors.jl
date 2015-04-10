using SGD
using Base.Test


p = LinearPredictor(3)
@test paramdim(p) == 3
@test sampledim(p) == 3
@test p([3, 4, 5], [6, 7, 8]) == 86
@test_throws DimensionMismatch p(zeros(4), zeros(4))
