
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
