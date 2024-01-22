import copy
import pickle
from typing import List
import numpy as np


def relu(x: np.array, derivative=False):
    if derivative:
        return np.where(x > 0, 1, 0)
    else:
        return np.where(x > 0, x, 0)


def sigmoid(x: np.array, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def tanh(x: np.array, derivative=False):
    if derivative:
        return 1 - np.power(x, 2)
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def softmax(x: np.array):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def NoneFunc( x , derivative= False):
    return x


class NeuralNetwork:

    def __init__(self, input_size):
        self.input_size = input_size
        self.layers: List[Layer] = []

    def add_layer(self, n=10, limits=(-0.5, 0.5), activation_function = None, weight: np.array = None):

        if not self.layers:
            shape = (n, self.input_size)
        else:
            shape = (n, self.layers[-1].shape[0])

        layer = Layer(shape=shape, limits=limits, activation_function=activation_function, weight=weight)

        self.layers.append(layer)

    def save_weights(self, filename='model_layers'):
        with open(f'{filename}.pkl', 'wb') as file:
            pickle.dump(self.layers, file)

    def load_weights(self, filename='model_layers'):
        with open(f'{filename}.pkl', 'rb') as file:
            self.layers = pickle.load(file)

    def calculate_error(self, output, target):
        return np.average(np.power(output - target, 2))

    def calculate_delta(self, output, target):
        return 2 / output.size * (output - target)

    def update_weights(self, layer, delta, alpha):
        layer.update_weight(layer.inputs, delta, alpha)

    def train_epoch(self, inputs, targets, learning_rate):


        for i in range(inputs.shape[1]):
            output = inputs[i].reshape(-1, 1)
            target = targets[i].reshape(-1, 1)

            for layer in self.layers:
                output = layer.forward_pass(output)

            output_delta = self.calculate_delta(output, target)

            next_weight = copy.deepcopy(self.layers[-1].weight)
            self.layers[-1].update_weight(output_delta, learning_rate)

            for layer in reversed(self.layers[0:-1]):
                hidden_delta = layer.backward_pass(output_delta, next_weight)
                output_delta = hidden_delta
                next_weight = copy.deepcopy(layer.weight)
                layer.update_weight(hidden_delta, learning_rate)


    def train(self, inputs, targets, learning_rate=0.01, epochs=50, mod=10 , filename = 'model_layers' ,savemod=100):

        for epoch in range(epochs + 1):

            self.train_epoch(inputs, targets, learning_rate)

            if (epoch + 1) % mod == 0:
                total_loss, acc = self.calculate_accuracy(inputs, targets)
                print(f"Epoch {epoch + 1}: Total Error = {total_loss:.10f} Acc:{acc:.02f}%")

            if (epoch + 1) % savemod == 0:

                self.save_weights(filename)
                print(f"Epoch {epoch + 1} SAVED")

        print("Training completed")

    def calculate_accuracy(self, inputs, targets, batch=1):

        accurate = 0
        error  = []

        for i in range(inputs.shape[1]):
            output = inputs[i].reshape(-1, 1)
            target = targets[i].reshape(-1, 1)

            for layer in self.layers:
                output = layer.forward_pass(output)

            predictions = np.equal(output.argmax(axis=0), target.argmax(axis=0))
            accurate += np.sum(predictions.astype(int) )

            err = self.calculate_error(output, target)
            error.append(err)

        return np.average(error), accurate / inputs.shape[1] * 100




class Layer:

    def __init__(self, shape=[10, 10], limits =(-0.5, 0.5), activation_function = None , weight: np.array = None):
        if activation_function is None :
            self.activation_func = NoneFunc
        else:
            self.activation_func = activation_function
        self.shape = shape

        if weight is None:
            self.weight = np.random.uniform(limits[0], limits[1], shape)
        else:
            self.weight = weight

        self.inputs = None
        self.output = None
        self.dropout_mask: np.array = None
        self.dropout = None

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.output = self.activation_func(np.dot(self.weight, inputs), derivative=False)

        return self.output

    def backward_pass(self, output_delta, next_layer = None):

        gradient = self.activation_func(self.output, derivative=True)
        delta = np.dot(next_layer.T, output_delta) * gradient
        return delta

    def update_weight(self, delta, alpha, batch_size=None):
        if batch_size is None:
            self.weight -= np.dot(delta, self.inputs.T) * alpha
        else:
            self.weight -= np.dot(delta, self.inputs.T) * alpha / batch_size
