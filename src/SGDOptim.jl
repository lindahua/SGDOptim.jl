module SGDOptim

using Compat
using ArrayViews

import Base: start, next, done
import Base.LinAlg: axpy!

export

    # streams.jl
    SampleStream,
    GenericSampleSeq,
    sample_seq,
    minibatch_seq,

    # loss.jl
    Loss,
    UnivariateLoss,
    MultinomialLoss,
    value_and_grad!,
    value_and_deriv,

    SqrLoss, sqrloss,
    HingeLoss, hingeloss,
    LogisticLoss, logisticloss,

    # callback.jl
    CallbackControl,
    NoCallback,
    ByInterval,

    simple_trace,
    gtcompare_trace,

    # sgd_std.jl
    sgd!,
    sgd


# source files

include("streams.jl")
include("predictors.jl")
include("loss.jl")
include("callback.jl")

include("sgdstd.jl")

end # module
