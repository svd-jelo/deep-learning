import numpy as np
from sklearn.metrics import accuracy_score
from src.functions import *

class Network:
    def __init__(self,
                 sizes,
                 activation_functions,
                 learning_rate=0.1,
                 mini_batch_size=10,
                 cost="mean_squared_error",
                 init_method="naive",
                 verbose=False,
                 random_state=None,
                 max_epochs = 50,
                 tol = 1e-3,
                 **kwargs):
        self.sizes = sizes
        self.learning_rate = learning_rate
        self.activation_functions = activation_functions
        self.mini_batch_size = mini_batch_size
        self.init_method = init_method
        self.verbose = verbose
        self.cost = cost
        self.max_epochs = max_epochs
        self.tol = tol

        if not random_state:
            random_state = 42
        self.random_state = random_state

        self.weights = None
        self.biases = None
        #self.activations = None
        self.kwargs = kwargs

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
        return {'dw': dw, 'db': db}, da_prev

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)

        # Initialize parameters
        self.weights, self.biases = map(list,
                                        zip(*Network.initialize_parameters(self.sizes,
                                                                           self.init_method,
                                                                           random_state=self.random_state,
                                                                           **self.kwargs)))

        #self.activations = []
        for epoch in range(self.max_epochs):
            perm = rng.permutation(X.shape[1])
            X_shuffled = X[:,perm]
            y_shuffled = y[:,perm]

            epoch_cost = 0
            for n in range(0,X_shuffled.shape[1],self.mini_batch_size):
                a_prev = X_shuffled[:,n:n+self.mini_batch_size]
                caches_forward = []
                for l in range(len(self.weights)):
                    cache = Network.feedforward(a_prev, self.weights[l], self.biases[l], activation=self.activation_functions[l])
                    caches_forward.append(cache)
                    a_prev = cache['a']
                    #self.activations.append(cache['a'])

                # Cost function
                cost_value = compute_cost(a_prev, y_shuffled[:,n:n+self.mini_batch_size], self.cost)
                epoch_cost += cost_value * self.mini_batch_size

                # Backward pass
                grads = []
                da_L = cost_backward(caches_forward[-1]['a'], y_shuffled[:,n:n+self.mini_batch_size], self.cost)
                if len(caches_forward) < 2:
                    a_prev = X[:,n:n+self.mini_batch_size]
                else:
                    a_prev = caches_forward[-2]['a']
                grads_L, da_prev = Network.backward(a_prev, da_L, caches_forward[-1], self.activation_functions[-1])
                grads.append(grads_L)

                for l in reversed(range(len(self.weights)-1)):
                    da = da_prev
                    a_prev = caches_forward[l-1]['a'] if l>0 else X_shuffled[:,n:n+self.mini_batch_size]
                    cache_forward = caches_forward[l]
                    activation_function = self.activation_functions[l]
                    grads_l, da_prev = Network.backward(a_prev, da, cache_forward, activation_function)
                    grads.append(grads_l)
                grads = grads[::-1]

                # Weight update
                assert len(grads) == len(self.weights)
                for l in range(len(grads)):
                    self.weights[l] -= self.learning_rate * grads[l]['dw']
                    self.biases[l] -= self.learning_rate * grads[l]['db']

            epoch_cost /= X_shuffled.shape[1]

            if self.verbose:
                if epoch % 100 == 0:
                    print(f'epoch: {epoch}, cost: {epoch_cost}')

            if abs(epoch_cost) < self.tol:
                print(f'epoch: {epoch}, cost: {epoch_cost}')
                break

        return self

    def predict_proba(self, X):
        caches_forward = []
        a_prev = X
        for l in range(len(self.weights)):
            cache = Network.feedforward(a_prev, self.weights[l], self.biases[l],
                                        activation=self.activation_functions[l])
            caches_forward.append(cache)
            a_prev = cache['a']

        return a_prev

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=0)

    def score(self, X, y):
        ypred = self.predict(X)
        return accuracy_score(y, ypred)


