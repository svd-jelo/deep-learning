import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from src.functions import *
from sklearn.base import BaseEstimator, TransformerMixin

class Network(BaseEstimator, TransformerMixin):
    def __init__(self,
                 sizes,
                 activation_functions,
                 learning_rate=0.1,
                 mini_batch_size=10,
                 cost="mean_squared_error",
                 init_method="naive",
                 weight_scale: float = 0.1,
                 bias_init = 0.001,
                 penalty=None,
                 alpha=0.1,
                 verbose=False,
                 random_state=None,
                 max_epochs = 50,
                 normalize=True,
                 early_stop_patience = np.inf,
                 learning_schedule_patience = np.inf,
                 min_delta = 1e-4,
                 momentum = 0.0,
                 tol = 1e-3,
                 monitor_cost=False,
                 monitor_accuracy=False,
                 val_data=None):
        self.sizes = sizes
        self.learning_rate = learning_rate
        self.activation_functions = activation_functions
        self.mini_batch_size = mini_batch_size
        self.init_method = init_method
        self.weight_scale = weight_scale
        self.bias_init = bias_init
        self.penalty = penalty
        self.alpha = alpha  #regularization parameter alpha
        self.verbose = verbose
        self.cost = cost
        self.max_epochs = max_epochs
        self.early_stop_patience = early_stop_patience
        self.learning_schedule_patience = learning_schedule_patience
        self.min_delta = min_delta
        self.momentum = momentum
        self.tol = tol
        self.monitor_cost = monitor_cost
        self.monitor_accuracy = monitor_accuracy
        self.random_state = random_state
        self.val_data = val_data
        self.normalize = normalize
        self.scale = None
        self.encode = None

        self.weights = None
        self.biases = None
        self.grads = None
        self.cost_value = None
        self.final_eta = None
        self.train_cost_cache = []
        self.val_cost_cache = []
        self.training_accuracy = []
        self.val_accuracy = []

    @staticmethod
    def initialize_parameters(layer_dims: list[int], method: str = "naive", random_state=None, weight_scale=0.1, bias_init=0.001):
        """
        initialize the weights of a feed-forward neural network given dimensions of each layer layer_dims, and method of initialization.
        if method is "naive", a naive initialization, using a standard normal distribution, is performed. If method is "he", the "He" initialization is performed.

        :param layer_dims: list of ints; dimensions of each layer, including input and output layers.
        :param method: str "naive" or "he"; default "naive"; if "naive", a standard normal distribution is used to
                        initialize the weights, and np.zeros for the biases; if "he", the "He" initialization is performed.
        :param random_state: int; random seed
        :param weight_scale: float; scaling factor for weight initialization
        :param bias_init: float; bias initialization
        :return: list of tuples (weights, biases) for each hidden layer.
        """
        L = len(layer_dims)
        parameters = []
        scale = weight_scale
        bias_init = bias_init

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

    def run_epoch(self, X, y):
        epoch_cost = 0
        for n in range(0,X.shape[1],self.mini_batch_size):
            a_prev = X[:,n:n+self.mini_batch_size]
            caches_forward = []
            for l in range(len(self.weights)):
                cache = Network.feedforward(a_prev, self.weights[l], self.biases[l], activation=self.activation_functions[l])
                caches_forward.append(cache)
                a_prev = cache['a']
                #self.activations.append(cache['a'])

            # Cost function
            cost_value = compute_cost(a_prev, y[:,n:n+self.mini_batch_size], self.cost)
            epoch_cost += cost_value * a_prev.shape[1]

            # Backward pass
            grads = []
            da_L = cost_backward(caches_forward[-1]['a'], y[:,n:n+self.mini_batch_size], self.cost)
            if len(caches_forward) < 2:
                a_prev = X[:,n:n+self.mini_batch_size]
            else:
                a_prev = caches_forward[-2]['a']
            grads_L, da_prev = Network.backward(a_prev, da_L, caches_forward[-1], self.activation_functions[-1])

            grads.append(grads_L)

            for l in reversed(range(len(self.weights)-1)):
                da = da_prev
                a_prev = caches_forward[l-1]['a'] if l>0 else X[:,n:n+self.mini_batch_size]
                cache_forward = caches_forward[l]
                activation_function = self.activation_functions[l]
                grads_l, da_prev = Network.backward(a_prev, da, cache_forward, activation_function)
                grads.append(grads_l)
            grads = grads[::-1]

            if self.grads is None:
                self.grads = grads
            for l, grad, grad_prev in zip(range(len(grads)), grads, self.grads):
                self.grads[l]['dw'] = -self.final_eta * grad['dw'] + self.momentum * grad_prev['dw']
                self.grads[l]['db'] = -self.final_eta * grad['db'] + self.momentum * grad_prev['db']

            # Weight update
            assert len(grads) == len(self.weights)
            for l in range(len(grads)):
                if self.penalty == 'l2':
                    self.weights[l] *= (1-self.alpha*self.final_eta/X.shape[1])
                elif self.penalty == 'l1':
                    mask = [self.weights[l] > 0, self.weights[l] < 0]
                    sign = [1,-1]
                    self.weights[l] -= self.final_eta * (self.alpha/X.shape[1]) * np.select(mask, sign, default=0)
                self.weights[l] += self.grads[l]['dw']
                self.biases[l] += self.grads[l]['db']

        epoch_cost /= X.shape[1]
        if self.penalty == 'l2':
            epoch_cost += 0.5 * (self.alpha / X.shape[1]) * sum(np.sum(np.square(w)) for w in self.weights)
        if self.penalty == 'l1':
            epoch_cost += (self.alpha / X.shape[1]) * sum(np.sum(np.abs(w)) for w in self.weights)

        return epoch_cost

    def fit(self, X, y):
        if self.random_state is None:
            self.random_state = 42

        # Training Data
        self.scale = StandardScaler()
        self.encode = LabelBinarizer()
        X_ = self.scale.fit_transform(X).T if self.normalize else X.copy().T
        y_ = self.encode.fit_transform(y).T

        # Validation Data
        X_val = None
        y_val = None
        if self.val_data is not None:
            X_val, y_val = self.val_data

        # Initialize parameters
        self.weights, self.biases = map(list,
                                        zip(*Network.initialize_parameters(self.sizes,
                                                                           self.init_method,
                                                                           random_state=self.random_state,
                                                                           weight_scale=self.weight_scale,
                                                                           bias_init=self.bias_init)))
        self.train_cost_cache = []
        self.val_cost_cache = []
        self.training_accuracy = []
        self.val_accuracy = []
        self.final_eta = self.learning_rate

        rng = np.random.default_rng(self.random_state)
        best_loss, val_loss, wait_sched, wait_stop, best_params = np.inf, np.nan, 0, 0, None
        for epoch in range(self.max_epochs):
            perm = rng.permutation(len(X))
            X_shuffled = X_[:,perm]
            y_shuffled = y_[:,perm]
            epoch_cost = self.run_epoch(X_shuffled, y_shuffled)
            self.cost_value = epoch_cost

            if self.monitor_accuracy:
                assert self.val_data is not None
                train_score = self.score(X, y)
                val_score = self.score(X_val, y_val)

                self.training_accuracy.append(train_score)
                self.val_accuracy.append(val_score)

            if self.verbose:
                if epoch % 100 == 0:
                    print(f'epoch: {epoch}, cost: {epoch_cost}')

            if self.monitor_cost:
                if self.val_data is not None:
                    a_val = self.predict_proba(X_val)
                    y_valt = self.encode.transform(y_val).T
                    val_loss = compute_cost(a_val, y_valt, self.cost)
                    self.val_cost_cache.append(val_loss)
                self.train_cost_cache.append(epoch_cost)

            # Early Stop
            if (self.early_stop_patience < np.iinfo(np.int64).max) or (self.learning_schedule_patience < np.iinfo(np.int64).max):
                assert self.val_data is not None, "No validation data passed"
                a_val = self.predict_proba(X_val)
                y_valt = self.encode.transform(y_val).T
                val_loss = compute_cost(a_val, y_valt, self.cost)

                if val_loss < best_loss - self.min_delta:
                    best_loss, wait = val_loss, 0
                    best_params = ([w.copy() for w in self.weights],
                                   [b.copy() for b in self.biases])
                else:
                    wait_sched += 1
                    wait_stop += 1
                    if wait_sched >= self.learning_schedule_patience:
                        if 128*self.final_eta <= self.learning_rate:
                            break
                        self.final_eta /= 2
                        wait_sched = 0
                    if wait_stop >= self.early_stop_patience:
                        break

            if abs(epoch_cost) < self.tol:
                print(f'epoch: {epoch}, cost: {epoch_cost}')
                break

        if best_params is not None:
            self.weights, self.biases = best_params

        return self

    def predict_proba(self, X):
        caches_forward = []
        Xt = self.scale.transform(X).T if self.normalize else X.copy().T
        a_prev = Xt
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




