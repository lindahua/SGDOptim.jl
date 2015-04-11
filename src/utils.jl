
default_lrate(t) = 1.0 / (1.0 + t)

half(x::Real) = 0.5 * x
half(x::Float32) = 0.5f0 * x
half(x::Float64) = 0.5 * x
