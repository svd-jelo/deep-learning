import numpy as np
from numpy import linalg as la

# INITIALIZATION METHODS
def he_initialization(dim, dim_prev, rng: np.random._generator.Generator, bias_init=0.001):
    """
    He Initialization for the parameters of a given layer. It is assumed that the activation function for the layer is ReLU.

    :param dim: int; dimension of the current layer
    :param dim_prev: int; dimension of previous layer.
    :param bias_init: float; initial value for the bias.
    :param rng: np.random._generator.Generator; random number generator
    :return: tuple; (weight, bias) where weights is a numpy array of shape (dim, dim_prev) and bias is a numpy array of shape (dim, 1).
    """
    epsilon = np.sqrt(2 / dim_prev)

    weights = rng.normal(loc=0, scale=epsilon, size=(dim, dim_prev))
    bias = np.ones(shape=(dim, 1)) * bias_init
    return weights, bias

def naive_initialization(dim, dim_prev, rng: np.random._generator.Generator, scale=0.01):
    """
    weights are initialized randomly from the standard normal distribution, scaled by `scale`, and has shape (dim, dim_prev);
    biases are initialized with zeros and has shape (dim, 1).

    :param dim: int; dimension of the current layer
    :param dim_prev: int; dimension of the previous layer
    :param scale: float; scaling factor for the weights
    :param rng: np.random._generator.Generator; random number generator
    :return: tuple (weights, biases) of np.ndarrays containing the initial weight and bias for the current layer
    """

    weights = rng.standard_normal(size=(dim, dim_prev)) * scale
    biases = np.zeros(shape=(dim, 1))

    return weights, biases

# ACTIVATION FORWARD
def linear_forward(a,w,b):
    """
    :param a: 2d np.ndarray representing the activations of the previous layer
    :param w: 2d np.ndarray representing the weights of the current layer; w.shape[1] must be equal to a.shape[0]
    :param b: 2d np.ndarray representing the bias of the current layer; b.shape[0] must be equal to w.shape[0]
    :return: np.ndarray of shape (w.shape[0],a.shape[1]); result of evaluating z = w @ a + b
    """
    assert len(a.shape) == 2 & len(b.shape) == 2 & len(w.shape) == 2
    assert a.shape[0] == w.shape[1] and w.shape[0] == b.shape[0]
    return w @ a + b

def relu_forward(z):
    """
    :param z: 2d np.ndarray representing the pre-activation of the current layer
    :return: np.ndarray; results of evaluating ReLU(z) = max(0,z)
    """
    assert len(z.shape) == 2
    return np.maximum(0,z)

def sigmoid_forward(z):
    """
    :param z: 2d np.ndarray representing the pre-activation of the current layer
    :return: np.ndarray; results of evaluating sigmoid(z)
    """
    assert len(z.shape) == 2
    return 1 / (1 + np.exp(-z))

def activation_forward(z, activation):
    """
    Utility function to compute the activation function given pre-activation values
    and type of activation function to be used.

    :param z: 2d np.ndarray representing the pre-activation value of the current layer
    :param activation: str; type of activation function to be used
    :return: 2d np.ndarray of shape z.shape representing the activation value
    """
    if activation == 'relu':
        return relu_forward(z)
    elif activation == 'sigmoid':
        return sigmoid_forward(z)
    else:
        raise ValueError('Activation function should be either relu or sigmoid')

# ACTIVATION BACKWARD
def relu_backward(da, z):
    """
    :param da: 2d np.ndarray representing the gradient with respect to the activation of the current layer
    :param z: 2d np.ndarray representing the pre-activation values of the current layer
    :return: 2d np.ndarray of shape z.shape representing the gradient with respect to the pre-activation
    """
    assert z.shape == da.shape
    u_z = np.where(z > 0, 1, 0)
    dz = da * u_z
    return dz

def sigmoid_backward(da, a):
    """
    :param da: 2d np.ndarray representing the gradient with respect to the activation of the current layer
    :param a: 2d np.ndarray representing the activation values of the current layer
    :return: 2d np.ndarray of shape a.shape representing the gradient with respect to the pre-activation
    """
    assert a.shape == da.shape
    return da * a * (1-a)

def activation_backward(da, z, a, activation):
    """
    :param da: 2d np.ndarray representing the gradient with respect to the activation of the current layer
    :param z: 2d np.ndarray representing the pre-activation values of the current layer
    :param a: 2d np.ndarray representing the activation values of the current layer
    :param activation: str; type of activation function to be used
    :return: 2d np.ndarray of shape a.shape representing the gradient with respect to the pre-activation
    """
    if activation == 'relu':
        return relu_backward(da, z)
    elif activation == 'sigmoid':
        return sigmoid_backward(da, a)
    else:
        raise ValueError('Activation function should be either relu or sigmoid')

# COST FUNCTIONS
def cross_entropy_multi(a,y):
    """
    :param a: 2d np.ndarray representing the activations from the output layer
    :param y: 2d np.ndarray representing ground truth labels
    :return: float; result of evaluating the cross entropy cost function C(a,y) for multiclass classification
    """
    assert a.shape == y.shape
    m = a.shape[1]
    cost = np.sum(y * np.log(a), axis=0, keepdims=True)
    cost = np.sum(cost, axis=1, keepdims=True)
    return (-1/m) * np.squeeze(cost)

def mean_squared_error(a,y):
    """
    :param a: 2d np.ndarray representing the activations from the output layer
    :param y: 2d np.ndarray representing ground truth labels
    :return: float; result of evaluating the mean squared error C(a,y)
    """
    assert a.shape == y.shape
    m = a.shape[1]
    cost = la.norm(a - y)**2
    return cost / (2*m)

def compute_cost(a,y,func):
    """
    :param a: 2d np.ndarray representing the activations from the output layer
    :param y: 2d np.ndarray representing ground truth labels
    :param func: str; type of cost function to be used
    :return: float; result of evaluating the mean squared error C(a,y)
    """
    if func == 'cross_entropy':
        return cross_entropy_multi(a, y)
    elif func == 'mean_squared_error':
        return mean_squared_error(a, y)
    else:
        raise ValueError('Cost function should be either cross_entropy or mean_squared_error')

# COST BACKWARD
def quadratic_cost_backward(a, y):
    """
    :param a: 2d np.ndarray representing the activations from the output layer
    :param y: 2d np.ndarray representing ground truth labels
    :return: 2d np.ndarray of shape a.shape representing da
    """
    assert a.shape == y.shape
    m = a.shape[1]
    return (1/m) * (a-y)

def cross_entropy_multi_backward(a, y):
    """
    :param a: 2d np.ndarray representing the activations from the output layer
    :param y: 2d np.ndarray representing ground truth labels
    :return: 2d np.ndarray of shape a.shape representing da
    """
    assert a.shape == y.shape
    m = a.shape[1]
    return -(1/m) * (y / a)

def cost_backward(a, y, func):
    """
    :param a: 2d np.ndarray representing the activations from the output layer
    :param y: 2d np.ndarray representing ground truth labels
    :param func: str; type of cost function to be used
    :return: 2d np.ndarray of shape a.shape representing da
    """
    if func == 'cross_entropy':
        return cross_entropy_multi_backward(a, y)
    elif func == 'mean_squared_error':
        return quadratic_cost_backward(a, y)
    else:
        raise ValueError('Cost function should be either cross_entropy or mean_squared_error')
