"""This file is WIP"""

class RegressionMetrics(object):
    """Metrics for regression approaches, using Theano and Lasagne."""

    def __init__(self, prediction, target):
        self.prediction = prediction
        self.target = target
