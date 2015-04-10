module SGD

using ArrayViews

import Base: call
import Base.LinAlg: axpy!

export

    # abstract types
    Loss,
    ScalarLoss,
    MultinomialLoss,

    # specific loss functors
    SqrLoss,
    sqrloss!,

    # solver
    sgd!,
    sgd,

    # utilities
    SGDRecord,
    CallbackControl,
    NoCallback,
    ByInterval,

    simple_trace,
    gtcompare_trace


include("common.jl")
include("loss.jl")
include("callback.jl")

include("sgdstd.jl")

end # module
