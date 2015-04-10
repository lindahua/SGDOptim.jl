# Standard implementation of SGD

function sgd!{T<:FloatingPoint}(loss!::ScalarLoss,
                                θ::DenseVector{T},
                                X::DenseMatrix{T},
                                y::AbstractVector,
                                order,
                                lrate,
                                cbctrl,
                                callback)

    # check dimensions
    n = size(X, 2)
    length(y) == n ||
        throw(DimensionMismatch("Inconsistent input diimensions."))

    # preparing storage
    g = similar(θ)
    tloss = 0.0

    # main loop
    for (t, i) in enumerate(order)
        v = loss!(g, θ, view(X, :, i), y[i])
        λ = lrate(t)
        axpy!(-λ, g, θ)  # θ <- θ - λ * g
        tloss += v

        cbctrl = check(cbctrl)
        if isready(cbctrl)
            callback(SGDRecord(t, t, v, tloss))
        end
    end

    # return the result
    return θ
end


function sgd{T<:FloatingPoint}(loss!::ScalarLoss,
                               θ::DenseVector{T},
                               X::DenseMatrix{T},
                               y::AbstractVector;
                               order=1:size(X,2),
                               lrate=t->1.0 / t,
                               cbctrl=NoCallback(),
                               callback=simple_trace)

    sgd!(loss!, copy(θ), X, y, order, lrate, cbctrl, callback)
end
