# Standard implementation of SGD

function sgd!{T<:FloatingPoint}(pred::UnivariatePredictor,
                                loss::UnivariateLoss,
                                reg::Regularizer,
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
    for s in stream
        t += 1
        n = nsamples(pred, θ, s...)

        # for loss
        v = loss_and_grad!(pred, loss, g, θ, s...)

        # for regularizer
        v += value_and_addgrad!(reg, g, θ)

        # update
        λ = convert(T, lrate(t))::T
        axpy!(-λ, g, θ)  # θ <- θ - λ * g

        # callback
        if cbinterval > 0 && t % cbinterval == 0
            callback(θ, t, n, v)
        end
    end

    # return the result
    return θ
end


function sgd{T<:FloatingPoint}(pred::UnivariatePredictor,
                               loss::UnivariateLoss,
                               θ::DenseVector{T},
                               stream::SampleStream;
                               reg::Regularizer=NoReg(),
                               lrate=default_lrate,
                               cbinterval::Int = 0,
                               callback=simple_trace)

    sgd!(pred, loss, reg, copy(θ), stream, lrate, cbinterval, callback)
end
