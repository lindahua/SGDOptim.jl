module SGD

import Base: call

export

    # abstract types
    Loss,
    ScalarLoss,
    MultinomialLoss,

    # specific loss functors
    SqrLoss,
    sqrloss!

include("loss.jl")

end # module
