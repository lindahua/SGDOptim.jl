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

    # predictors.jl
    Predictor,
    UnivariatePredictor,
    MultivariatePredictor,

    LinearPredictor, linear_predictor,

    # loss.jl
    Loss,
    UnivariateLoss,
    MultinomialLoss,
    loss_and_grad!,
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
