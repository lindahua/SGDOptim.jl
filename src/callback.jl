# Callback controls

abstract CallbackControl

type NoCallback <: CallbackControl
end

check(c::NoCallback) = c
isready(c::NoCallback) = false


type ByInterval <: CallbackControl
    intval::Int
    r::Int

    function ByInterval(n::Int)
        n > 0 || error("The interval must be positive.")
        new(n, 0)
    end
end

function check(c::ByInterval)
    if c.r > 0
        c.r -= 1
    else
        c.r = c.intval - 1
    end
    c
end
isready(c::ByInterval) = (c.r == 0)


# Callbacks

function simple_trace(θ::AbstractArray, t::Int, n::Int, v::Real)
    @printf("Iter %d: avg.loss = %.4e\n",
        t, v / n)
end

function gtcompare_trace(θg::AbstractArray)
    function _trace(θ::AbstractArray, t::Int, n::Int, v::Real)
        dev = vecnorm(θ - θg)
        @printf("Iter %d: avg.loss = %.4e, deviation = %.4e\n",
            t, v / n, dev)
    end
    return _trace
end
