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
    CallbackControl,
    NoCallback,
    ByInterval,

    simple_trace


include("common.jl")
include("utils.jl")
include("loss.jl")
include("sgdstd.jl")

end # module
