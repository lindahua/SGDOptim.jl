

# AbstractLoss subsumes all kinds of loss functions
abstract Loss

abstract ScalarLoss <: Loss
abstract MultinomialLoss <: Loss
