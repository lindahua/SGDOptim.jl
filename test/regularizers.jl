using SGDOptim
using Base.Test


θ = [4.0, -5.0, 6.0]
g0 = [1.0, 2.0, 3.0]


# NoReg

g = copy(g0)
v = SGDOptim.value_and_addgrad!(NoReg(), g, θ)
@test v == 0.0
@test g == g0

# SqrL2Reg

g = copy(g0)
v = SGDOptim.value_and_addgrad!(SqrL2Reg(2.5), g, θ)
@test v == 96.25
@test g == g0 + 2.5θ

# L1Reg

g = copy(g0)
v = SGDOptim.value_and_addgrad!(L1Reg(2.5), g, θ)
@test v == 37.5
@test g == g0 + 2.5 * sign(θ)
