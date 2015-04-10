
type SGDRecord
    niters::Int
    nsamples::Int
    loss::Float64
    total_loss::Float64

    SGDRecord(t::Int, ns::Int, loss::Float64, tloss::Float64) = new(t, ns, loss, tloss)
end

current_loss(r::SGDRecord) = r.loss
avg_loss(r::SGDRecord) = r.total_loss / r.nsamples
