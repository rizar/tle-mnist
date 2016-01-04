#!/usr/bin/env python

import logging
from argparse import ArgumentParser

from theano import tensor

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


def main(save_to, num_epochs):
    mlp = MLP([None], [784, 10],
              weights_init=IsotropicGaussian(0.01),
              biases_init=Constant(0))
    mlp.initialize()
    x = tensor.matrix('features')
    y = tensor.lmatrix('targets')
    scores = mlp.apply(x)

    target_scores = tensor.set_subtensor(
        tensor.zeros((y.shape[0], 10))[tensor.arange(y.shape[0]), y.flatten()],
        1)
    score_diff = scores - target_scores

    # Logistic Regression
    # cost = Softmax().categorical_cross_entropy(y.flatten(), scores).mean()

    # MSE
    # cost = ((scores - target_scores) ** 2).mean()

    # Perceptron
    cost = (scores.max(axis=1) - scores[tensor.arange(y.shape[0]), y.flatten()]).mean()

    # TLE
    # cost = abs(score_diff[tensor.arange(y.shape[0]), y.flatten()]).mean()
    # cost += abs(score_diff[tensor.arange(y.shape[0]), scores.argmax(axis=1)]).mean()

    # TLEcut
    # Score of the groundtruth should be greater or equal than its target score
    # cost = tensor.maximum(0, -score_diff[tensor.arange(y.shape[0]), y.flatten()]).mean()
    # Score of the prediction should be less or equal than its actual score
    # cost += tensor.maximum(0, score_diff[tensor.arange(y.shape[0]), scores.argmax(axis=1)]).mean()

    # TLE2
    # score_diff = scores - target_scores
    # cost = ((score_diff[tensor.arange(y.shape[0]), y.flatten()]) ** 2).mean()
    # cost += ((score_diff[tensor.arange(y.shape[0]), scores.argmax(axis=1)]) ** 2).mean()

    # Direct loss minimization
    # epsilon = 0.1
    # cost = (- scores[tensor.arange(y.shape[0]), (scores + epsilon * target_scores).argmax(axis=1)]
    #         + scores[tensor.arange(y.shape[0]), scores.argmax(axis=1)]).mean()
    # cost /= epsilon

    error_rate = MisclassificationRate().apply(y.flatten(), scores)

    cg = ComputationGraph([cost])
    cost.name = 'final_cost'

    mnist_train = MNIST(("train",))
    mnist_test = MNIST(("test",))

    rule = Momentum(learning_rate=0.00001, momentum=0.999)
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
                  CallbackExtension(
                      lambda: rule.learning_rate.set_value(rule.learning_rate.get_value() * 0.9),
                      after_epoch=True),
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
                ['test_final_cost',
                 'test_misclassificationrate_apply_error_rate'],
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of training an MLP on"
                            " the MNIST dataset.")
    parser.add_argument("--num-epochs", type=int, default=2,
                        help="Number of training epochs to do.")
    parser.add_argument("save_to", default="mnist.pkl", nargs="?",
                        help=("Destination to save the state of the training "
                              "process."))
    args = parser.parse_args()
    main(args.save_to, args.num_epochs)
