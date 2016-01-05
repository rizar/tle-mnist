#!/usr/bin/env python

import sys
import logging
import numpy
import json
from argparse import ArgumentParser

from theano import tensor
import pandas

from blocks.algorithms import GradientDescent, Scale, Momentum
from blocks.bricks import MLP, Tanh, Softmax, WEIGHT
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.initialization import IsotropicGaussian, Constant
from fuel.streams import DataStream
from fuel.transformers import Flatten
from fuel.datasets import MNIST
from fuel.schemes import SequentialScheme
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint, SimpleExtension
from blocks.extensions.monitoring import (DataStreamMonitoring,
                                          TrainingDataMonitoring)
from blocks.main_loop import MainLoop

try:
    from blocks.extras.extensions.plot import Plot
    BLOCKS_EXTRAS_AVAILABLE = True
except:
    BLOCKS_EXTRAS_AVAILABLE = False


class CallbackExtension(SimpleExtension):

    def __init__(self, callback, *args, **kwargs):
        self.callback = callback
        super(CallbackExtension, self).__init__(*args, **kwargs)

    def do(self, *args):
        self.callback()


def main(save_to, cost_name, learning_rate, momentum, num_epochs):
    mlp = MLP([None], [784, 10],
              weights_init=IsotropicGaussian(0.01),
              biases_init=Constant(0))
    mlp.initialize()
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')
    scores = mlp.apply(x)

    batch_size = y.shape[0]
    indices = tensor.arange(y.shape[0])
    target_scores = tensor.set_subtensor(
        tensor.zeros((batch_size, 10))[indices, y.flatten()],
        1)
    score_diff = scores - target_scores

    # Logistic Regression
    if cost_name == 'lr':
        cost = Softmax().categorical_cross_entropy(y.flatten(), scores).mean()
    # MSE
    elif cost_name == 'mse':
        cost = (score_diff ** 2).mean()
    # Perceptron
    elif cost_name == 'perceptron':
        cost = (scores.max(axis=1) - scores[indices, y.flatten()]).mean()
    # TLE
    elif cost_name == 'minmin':
        cost = abs(score_diff[indices, y.flatten()]).mean()
        cost += abs(score_diff[indices, scores.argmax(axis=1)]).mean()
    # TLEcut
    elif cost_name == 'minmin_cut':
        # Score of the groundtruth should be greater or equal than its target score
        cost = tensor.maximum(0, -score_diff[indices, y.flatten()]).mean()
        # Score of the prediction should be less or equal than its actual score
        cost += tensor.maximum(0, score_diff[indices, scores.argmax(axis=1)]).mean()
    # TLE2
    elif cost_name == 'minmin2':
        cost = ((score_diff[tensor.arange(y.shape[0]), y.flatten()]) ** 2).mean()
        cost += ((score_diff[tensor.arange(y.shape[0]), scores.argmax(axis=1)]) ** 2).mean()
    elif cost_name == 'direct':
        # Direct loss minimization
        epsilon = 0.1
        cost = (- scores[tensor.arange(y.shape[0]), (scores + epsilon * target_scores).argmax(axis=1)]
                + scores[tensor.arange(y.shape[0]), scores.argmax(axis=1)]).mean()
        cost /= epsilon
    else:
        raise ValueError("Unknown cost " + cost)

    error_rate = MisclassificationRate().apply(y.flatten(), scores)
    error_rate.name = 'error_rate'

    cg = ComputationGraph([cost])
    cost.name = 'cost'

    mnist_train = MNIST(("train",))
    mnist_test = MNIST(("test",))

    if learning_rate == None:
        learning_rate = 0.0001
    if momentum == None:
        momentum = 0.99
    rule = Momentum(learning_rate=learning_rate,
                    momentum=momentum)
    algorithm = GradientDescent(
        cost=cost, parameters=cg.parameters,
        step_rule=rule)
    extensions = [Timing(),
                  FinishAfter(after_n_epochs=num_epochs),
                  DataStreamMonitoring(
                      [cost, error_rate],
                      Flatten(
                          DataStream.default_stream(
                              mnist_test,
                              iteration_scheme=SequentialScheme(
                                  mnist_test.num_examples, 500)),
                          which_sources=('features',)),
                      prefix="test"),
                  # CallbackExtension(
                  #    lambda: rule.learning_rate.set_value(rule.learning_rate.get_value() * 0.9),
                  #    after_epoch=True),
                  TrainingDataMonitoring(
                      [cost, error_rate,
                       aggregation.mean(algorithm.total_gradient_norm),
                       rule.learning_rate],
                      prefix="train",
                      after_epoch=True),
                  Checkpoint(save_to),
                  Printing()]

    if BLOCKS_EXTRAS_AVAILABLE:
        extensions.append(Plot(
            'MNIST example',
            channels=[
                ['test_cost',
                 'test_error_rate'],
                ['train_total_gradient_norm']]))

    main_loop = MainLoop(
        algorithm,
        Flatten(
            DataStream.default_stream(
                mnist_train,
                iteration_scheme=SequentialScheme(
                    mnist_train.num_examples, 50)),
            which_sources=('features',)),
        model=Model(cost),
        extensions=extensions)

    main_loop.run()

    df = pandas.DataFrame.from_dict(main_loop.log, orient='index')
    res = {'cost' : cost_name,
           'learning_rate' : learning_rate,
           'momentum' : momentum,
           'train_cost' : df.train_cost.iloc[-1],
           'test_cost' : df.test_cost.iloc[-1],
           'best_test_cost' : df.test_cost.min(),
           'train_error' : df.train_error_rate.iloc[-1],
           'test_error' : df.test_error_rate.iloc[-1],
           'best_test_error' : df.test_error_rate.min()}
    res = {k: float(v) if isinstance(v, numpy.ndarray) else v for k, v in res.items()}
    json.dump(res, sys.stdout)
    sys.stdout.flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training an MLP on"
                            " the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=2,
                        help="Number of training epochs to do.")
    parser.add_argument("--cost", type=str, help="Cost type")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--momentum", type=float, help="Momentum")
    parser.add_argument("--grid-search", action="store_true",
                        default=False, help="Do grid search")
    parser.add_argument("save_to", default="mnist.pkl", nargs="?",
                        help=("Destination to save the state of the training "
                              "process."))
    args = parser.parse_args()

    if not args.grid_search:
        main(args.save_to, args.cost, args.learning_rate, args.momentum, args.num_epochs)
    else:
        if args.learning_rate or args.momentum:
            raise ValueError("If you want grid search, do not specify hyperparameters")
        for lr in [0.000001, 0.00001, 0.0001, 0.001, 0.01]:
            for mom in [0, 0.9, 0.99, 0.999]:
                if lr / mom < 0.009:
                    main("{}_{}_{}.zip".format(args.save_to, lr, mom), args.cost, lr, mom, args.num_epochs)

