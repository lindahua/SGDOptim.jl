# Standard implementation of SGD

# compute the objective value with both loss and regularization
# and also compute the gradient (writing to g)
#
function value_and_grad!(pred::Predictor,
                         loss::Loss,
                         reg::Regularizer,
                         g::AbstractArray,
                         θ::AbstractArray,
                         s)

    v = value_and_grad!(pred, loss, g, θ, s...)
    v += value_and_addgrad!(reg, g, θ)
    return v
end

function sgd!{T<:FloatingPoint}(pred::Predictor, loss::Loss, reg::Regularizer,
                                θ::DenseVecOrMat{T}, stream::SampleStream,
                                lrate, cbinterval::Int, callback)

    # preparing storage
    g = similar(θ)
    tloss = 0.0

    # main loop
    t = 0
    for s in stream
        t += 1
        n = nsamples(pred, θ, s...)

        # evaluate objective and gradient
        v = value_and_grad!(pred, loss, reg, g, θ, s)

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


function sgd{T<:FloatingPoint}(pred::Predictor,
                               loss::Loss,
                               θ::DenseVecOrMat{T},
                               stream::SampleStream;
                               reg::Regularizer=NoReg(),
                               lrate=default_lrate,
                               cbinterval::Int = 0,
                               callback=simple_trace)

    sgd!(pred, loss, reg, copy(θ), stream, lrate, cbinterval, callback)
end
