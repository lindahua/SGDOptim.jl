module SGDOptim

using Compat
using Reexport
using ArrayViews
@reexport using EmpiricalRisks

import Base: start, next, done
import Base.LinAlg: axpy!

export
    # from ArrayViews
    views,

    # streams.jl
    SampleStream,
    GenericSampleSeq,
    sample_seq,
    minibatch_seq,

    # callback.jl
    simple_trace,
    gtcompare_trace,

    # sgd_std.jl
    sgd!,
    sgd


# source files

include("utils.jl")
include("streams.jl")
include("callback.jl")

include("sgdstd.jl")

end # module
