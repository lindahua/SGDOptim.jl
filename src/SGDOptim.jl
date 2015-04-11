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
    predict,
    nsamples,

    LinearPredictor,
    AffinePredictor,

    # loss.jl
    Loss,
    UnivariateLoss,
    MultinomialLoss,

    SqrLoss,
    HingeLoss,
    LogisticLoss,

    # regularizers.jl
    Regularizer,
    NoReg,
    SqrL2Reg,
    L1Reg,
    ElasticReg,

    # callback.jl
    simple_trace,
    gtcompare_trace,

    # sgd_std.jl
    sgd!,
    sgd


# source files

include("utils.jl")
include("streams.jl")
include("predictors.jl")
include("loss.jl")
include("regularizers.jl")
include("callback.jl")

include("sgdstd.jl")

end # module
