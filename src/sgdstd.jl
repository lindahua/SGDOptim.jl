# Standard implementation of SGD

function value_and_grad!{T<:FloatingPoint}(rmodel::RiskModel, g::StridedArray{T}, θ::StridedArray{T}, x, y)
    v_risk, _ = value_and_addgrad!(rmodel, 0, g, 1, θ, x, y)
    v_regr, _ = value_and_addgrad!(rmodel, 1, g, 1, θ, x, y)
    return v_risk + v_regr
end



# Standard SGD: implementation
function sgd!{T<:FloatingPoint}(rmodel::SupervisedRiskModel, reg::Regularizer,
                                θ::StridedArray{T}, stream::SampleStream,
                                lrate, cbinterval::Int, callback)

    # preparing storage
    g = similar(θ)
    tloss = 0.0

    # main loop
    pm = rmodel.predmodel
    t = 0
    for (x, y) in stream
        t += 1
        n = ninputs(pm, x)

        # evaluate objective and gradient
        v = value_and_grad!(rmodel, g, θ, x, y)

        # update
        λ = convert(T, lrate(t))
        axpy!(-λ, g, θ)  # θ <- θ - λ * g

        # callback
        if cbinterval > 0 && t % cbinterval == 0
            callback(θ, t, n, v)
        end
    end

    # return the result
    return θ
end

# Standard SGD: facet
function sgd{T<:FloatingPoint}(rmodel::SupervisedRiskModel,
                               θ::StridedArray{T},
                               stream::SampleStream;
                               reg::Regularizer=ZeroReg(),
                               lrate=default_lrate,
                               cbinterval::Int = 0,
                               callback=simple_trace)

    sgd!(rmodel, reg, copy(θ), stream, lrate, cbinterval, callback)
end
