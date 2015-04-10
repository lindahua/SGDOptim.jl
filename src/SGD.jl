module SGD

using ArrayViews

import Base: start, next, done, call
import Base.LinAlg: axpy!

export

    # streams.jl
    SampleStream,
    SampleSeq,

    # loss.jl
    Loss,
    ScalarLoss,
    MultinomialLoss,

    SqrLoss,
    sqrloss!,

    # utils.jl
    SGDRecord,
    CallbackControl,
    NoCallback,
    ByInterval,

    simple_trace,
    gtcompare_trace,

    # sgd_std.jl
    sgd!,
    sgd

include("common.jl")
include("streams.jl")
include("loss.jl")
include("callback.jl")

include("sgdstd.jl")

end # module
