module SGD

import Base: call

export

    # types
    Loss,
    ScalarLoss,
    MultinomialLoss,

    Predictor,
    LinearPredictor,

    # methods
    paramdim,
    sampledim


include("common.jl")
include("predictors.jl")

end # module
