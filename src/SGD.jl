module SGD

using ArrayViews

import Base: start, next, done, call
import Base.LinAlg: axpy!

export

    # streams.jl
    SampleStream,
    GenericSampleSeq,
    sample_seq,
    minibatch_seq,

    # loss.jl
    Loss,
    ScalarLoss,
    MultinomialLoss,

    SqrLoss, sqrloss!,
    HingeLoss, hingeloss!,
    LogisticLoss, logisticloss!,    

    # callback.jl
    CallbackControl,
    NoCallback,
    ByInterval,

    simple_trace,
    gtcompare_trace,

    # sgd_std.jl
    sgd!,
    sgd

include("streams.jl")
include("loss.jl")
include("callback.jl")

include("sgdstd.jl")

end # module
