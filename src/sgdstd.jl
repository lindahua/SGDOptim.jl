# Standard implementation of SGD

function sgd!{T<:FloatingPoint}(loss::UnivariateLoss,
                                θ::DenseVector{T},
                                stream::SampleStream,
                                lrate,
                                cbinterval::Int,
                                callback)

    # preparing storage
    g = similar(θ)
    tloss = 0.0

    # main loop
    t = 0
    for (inds, s) in stream
        t += 1
        n = length(inds)

        v = value_and_grad!(loss, g, θ, s...)
        λ = convert(T, lrate(t))::T
        axpy!(-λ, g, θ)  # θ <- θ - λ * g

        if cbinterval > 0 && t % cbinterval == 0
            callback(θ, t, n, v)
        end
    end

    # return the result
    return θ
end


function sgd{T<:FloatingPoint}(loss!::UnivariateLoss,
                               θ::DenseVector{T},
                               stream::SampleStream;
                               lrate=t->1.0 / t,
                               cbinterval::Int = 0,
                               callback=simple_trace)

    sgd!(loss!, copy(θ), stream, lrate, cbinterval, callback)
end
