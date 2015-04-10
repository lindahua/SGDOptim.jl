
type SGDRecord
    sol::DenseVector
    niters::Int
    nsamples::Int
    total_loss::Float64

    SGDRecord(sol::DenseVector, t::Int, ns::Int, tloss::Float64) =
        new(sol, t, ns, tloss)
end

avg_loss(r::SGDRecord) = r.total_loss / r.nsamples
