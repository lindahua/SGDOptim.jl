# Standard implementation of SGD

function sgd!{T<:FloatingPoint}(loss!::ScalarLoss,
                                θ::DenseVector{T},
                                stream::SampleStream,
                                lrate,
                                cbctrl,
                                callback)

    # preparing storage
    g = similar(θ)
    tloss = 0.0

    # main loop
    t = 0
    for s in stream
        t += 1
        v = loss!(g, θ, s...)
        λ = convert(T, lrate(t))::T
        axpy!(-λ, g, θ)  # θ <- θ - λ * g
        tloss += v

        cbctrl = check(cbctrl)
        if isready(cbctrl)
            callback(SGDRecord(θ, t, t, tloss))
        end
    end

    # return the result
    return θ
end


function sgd{T<:FloatingPoint}(loss!::ScalarLoss,
                               θ::DenseVector{T},
                               stream::SampleStream;
                               lrate=t->1.0 / t,
                               cbctrl=NoCallback(),
                               callback=simple_trace)

    sgd!(loss!, copy(θ), stream, lrate, cbctrl, callback)
end
