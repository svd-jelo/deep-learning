import numpy as np
from src.functions import *

class Network:
    def __init__(self,
                 sizes,
                 learning_rate=0.1,
                 mini_batch_size=10,
                 cost="quadratic",
                 init_method="naive",
                 verbose=False,
                 random_state=None,
                 **kwargs):
        self.sizes = sizes
        self.learning_rate = learning_rate
        self.mini_batch_size = mini_batch_size
        self.init_method = init_method
        self.verbose = verbose
        self.cost = cost

        if not random_state:
            random_state = 42
        self.random_state = random_state

        self.weights, self.biases = map(list, zip(*Network.initialize_parameters(self.sizes, self.init_method, random_state=random_state, **kwargs)))

    @staticmethod
    def initialize_parameters(layer_dims: list[int], method: str = "naive", random_state=None, **kwargs):
        """
        initialize the weights of a feed-forward neural network given dimensions of each layer layer_dims, and method of initialization.
        if method is "naive", a naive initialization, using a standard normal distribution, is performed. If method is "he", the "He" initialization is performed.

        :param layer_dims: list of ints; dimensions of each layer, including input and output layers.
        :param method: str "naive" or "he"; default "naive"; if "naive", a standard normal distribution is used to
                        initialize the weights, and np.zeros for the biases; if "he", the "He" initialization is performed.
        :param random_state: int; random seed
        :return: list of tuples (weights, biases) for each hidden layer.
        """
        L = len(layer_dims)
        parameters = []
        scale = kwargs.get("scale", 0.01)
        bias_init = kwargs.get("bias_init", 0.001)

        if not random_state:
            random_state = 42
        rng = np.random.default_rng(random_state)

        # remaining hidden layers
        for l in range(1, L):
            dim_prev = layer_dims[l-1]
            dim = layer_dims[l]

            if method == "naive":
                weights, biases = naive_initialization(dim, dim_prev, rng=rng, scale=scale)
            else:
                weights, biases = he_initialization(dim, dim_prev, bias_init=bias_init, rng=rng)

            parameters.append((weights, biases))

        return parameters

    @staticmethod
    def feedforward(a_prev, weight, bias, activation="sigmoid"):
        """
        Function to run forward pass using sigmoid as the activation function.

        :param a_prev: 2d np.ndarray representing the activation of the previous layer
        :param weight: 2d np.ndarrays representing the weight of the given layer
        :param bias: 2d np.ndarrays representing the bias of the given layer
        :param activation: str "sigmoid" or "relu"; default "sigmoid
        :return: dict[str, np.ndarray] containing a, z, weight, and bias
        """
        assert a_prev.shape[0] == weight.shape[1] and bias.shape[0] == weight.shape[0]
        z = linear_forward(a_prev, weight, bias)
        a = activation_forward(z, activation)
        cache_forward = {'a': a, 'z': z, 'weight': weight, 'bias': bias}
        return cache_forward

    @staticmethod
    def backward(a_prev, da, cache_forward, activation="sigmoid"):
        """
        Function to run backward pass given cached values from the forward pass on the current layer,
        gradient with respect to the activation of the current layer, and activation values of the
        previous layer.

        :param a_prev: np.ndarray representing the activation values of the previous layer
        :param da: gradient with respect to the activations of the current layer
        :param cache_forward: dict[str, np.ndarray] containing the cached values from the forward pass on the current layer
        :param activation: str "sigmoid" or "relu"; default "sigmoid
        :return: dict[str, np.ndarray] containing the gradients with respect to the parameters of the current layer
                and gradient with respect to the previous activation
        """
        a = cache_forward['a']
        z = cache_forward['z']
        weight = cache_forward['weight']

        dz = activation_backward(da, z, a, activation)
        dw = dz @ a_prev.T
        db = np.sum(dz, axis=1, keepdims=True)
        da_prev = weight.T @ dz
        grads = {'dw': dw, 'db': db, 'da_prev': da_prev}
        return grads

    def fit(self, X, y):
        



