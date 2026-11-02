import json

import numpy as np
from sklearn.metrics import accuracy_score
from src.functions import *
from sklearn.base import BaseEstimator, TransformerMixin
import torch
from torch.nn.functional import sigmoid, conv2d, max_pool2d, softmax, relu

class FullyConnectedLayer:
    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0, random_state=None):
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout

        self.init_std = 1.0
        if self.activation_fn.__name__ == 'sigmoid':
            self.init_std = np.sqrt(1.0/n_in)
        elif self.activation_fn.__name__ == 'relu':
            self.init_std = np.sqrt(2.0/n_in)

        # Initialize weight and biases
        seed = random_state if not random_state is None else 42
        self.rng = torch.Generator(device="mps").manual_seed(seed)
        self.w = torch.normal(mean=0.0, std=self.init_std, size=(n_in, n_out), generator=self.rng, requires_grad=True, device='mps')
        self.b = torch.normal(mean=0.0, std=1.0, size=(n_out,), generator=self.rng, requires_grad=True, device='mps')
        self.params = [self.w, self.b]

        #Attributes to be defined later
        self.inpt = None
        self.output = None
        self.y_out = None
        self.inpt_dropout = None
        self.output_dropout = None

    def train_forward(self, inpt_dropout, mini_batch_size):
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape((mini_batch_size, self.n_in)), self.p_dropout, self.rng)
        self.output_dropout = self.activation_fn(torch.matmul(self.inpt_dropout, self.w) + self.b)

    def forward(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.output = self.activation_fn((1 - self.p_dropout) * torch.matmul(self.inpt, self.w) + self.b)
        self.y_out = torch.argmax(self.output, dim=1)

    def accuracy(self,y):
        return torch.mean(torch.eq(y, self.y_out))

class ConvPoolLayer:
    def __init__(self, filter_shape, image_shape, activation_fn=sigmoid, pool_size=(2,2), stride=1, random_state=None):
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.activation_fn = activation_fn
        self.pool_size = pool_size
        self.stride = stride

        # Initialize weight and biases
        seed = random_state if not random_state is None else 42
        self.rng = torch.Generator(device="mps").manual_seed(seed)
        self.w = torch.normal(mean=0.0, std=np.sqrt(1.0/(filter_shape[1] * np.prod(filter_shape[2:]))), size=filter_shape, generator=self.rng, requires_grad=True, device='mps')
        self.b = torch.normal(mean=0.0, std=1.0, size=(filter_shape[0],), generator=self.rng, requires_grad=True, device='mps')
        self.params = [self.w, self.b]

        # attributes to be defined later
        self.inpt = None
        self.output = None
        self.output_dropout = None

    def forward(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape(mini_batch_size,*self.image_shape)
        conv_out = conv2d(self.inpt, self.w, self.b, stride=self.stride)
        activation_out = self.activation_fn(conv_out)
        self.output = max_pool2d(activation_out, self.pool_size)
        self.output_dropout = self.output #no dropout

class SoftMaxLayer:
    def __init__(self, n_in, n_out, p_dropout=0, random_state=None):
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout

        # Initialize weights and biases
        seed = random_state if not random_state is None else 42
        self.rng = torch.Generator(device="mps").manual_seed(seed)
        self.w = torch.zeros((n_in, n_out), requires_grad=True, device='mps')
        self.b = torch.zeros((n_out,), requires_grad=True, device='mps')
        self.params = [self.w, self.b]

        # attributes to be defined later
        self.inpt = None
        self.logits = None
        self.output = None
        self.log_output = None
        self.y_out = None
        self.inpt_dropout = None
        self.output_dropout = None
        self.logits_dropout = None
        self.log_output_dropout = None

    def train_forward(self, inpt_dropout, mini_batch_size):
        self.inpt_dropout = dropout_layer(inpt_dropout.reshape(mini_batch_size, self.n_in), self.p_dropout, self.rng)
        self.logits_dropout = torch.matmul(self.inpt_dropout, self.w) + self.b
        self.log_output_dropout = torch.log_softmax(self.logits_dropout, dim=1)

    def forward(self, inpt, mini_batch_size):
        self.inpt = inpt.reshape((mini_batch_size, self.n_in))
        self.logits = (1-self.p_dropout)*torch.matmul(self.inpt, self.w) + self.b
        self.output = softmax(self.logits, dim=1)
        self.y_out = torch.argmax(self.output, dim=1)
        self.log_output = torch.log_softmax(self.logits, dim=1)

    def cost(self, y, mode='train'):
        if mode == 'train':
            return -(y * self.log_output_dropout).sum(dim=1).mean()
        elif mode == 'validate':
            return -(y * self.log_output).sum(dim=1).mean()

    def accuracy(self, y):
        return torch.mean(torch.eq(y, self.y_out).float())

class NeuralNet:
    def __init__(self, layers):
        self.layers = layers
        self.params = [param for layer in self.layers for param in layer.params]

    def sgd(self, training_data, validation_data, test_data, epochs, minibatch_size, eta, lmbda=0.0, random_state=None):
        # prepare training data
        X_train, y_train = training_data
        num_training_batches = len(X_train) // minibatch_size
        num_training_batches += 1 if len(X_train) % minibatch_size != 0 else 0

        # prepare validation data
        X_val, y_val = validation_data
        num_validation_batches = len(X_val) // minibatch_size
        num_validation_batches += 1 if len(X_val) % minibatch_size != 0 else 0
        validation_costs = []
        validation_scores = []

        # prepared test data
        X_test, y_test = test_data
        num_test_batches = len(X_test) // minibatch_size
        num_test_batches += 1 if len(X_test) % minibatch_size != 0 else 0
        test_score = 0.0

        l2_norm_squared = 0.0
        iteration = 0
        best_validation_accuracy = 0.0
        rng = torch.Generator().manual_seed(random_state)
        for epoch in range(epochs):
            perm = torch.randperm(len(X_train), generator=rng)
            X_train_shuff = X_train[perm, :]
            y_train_shuff = y_train[perm, :]
            validation_accuracy = 0.0
            validation_cost = 0.0

            # TRAIN
            for i in range(num_training_batches):
                iteration = num_training_batches*epoch + i
                if iteration % 1000 == 0:
                    print(f"Training mini-batch number {iteration}")

                X_train_mb = X_train_shuff[i * minibatch_size:(i + 1) * minibatch_size]
                y_train_mb = y_train_shuff[i * minibatch_size:(i + 1) * minibatch_size]

                # Zero-out gradients
                for layer in self.layers:
                    if not layer.w.grad is None:
                        layer.w.grad.zero_()
                    if not layer.b.grad is None:
                        layer.b.grad.zero_()

                # Forward
                activation = X_train_mb
                for layer in self.layers:
                    if isinstance(layer, ConvPoolLayer):
                        layer.forward(activation, len(X_train_mb))
                    else:
                        layer.train_forward(activation, len(X_train_mb))
                    activation = layer.output_dropout

                if lmbda > 0.0:
                    l2_norm_squared = 0.5 * lmbda * sum((layer.w ** 2).sum() for layer in self.layers)

                training_cost_mb = self.layers[-1].cost(y_train_mb) + l2_norm_squared * len(X_train_mb) / len(X_train)  # cost_ij

                if not torch.isfinite(training_cost_mb).all():
                    print(f'WARNING: Cost is non-finite at iteration {iteration} with max logit {torch.max(torch.abs(self.layers[-1].logits_dropout))}')

                # Backward
                training_cost_mb.backward()

                # Update
                with torch.no_grad():
                    for layer in self.layers:
                        layer.w -= eta * layer.w.grad
                        layer.b -= eta * layer.b.grad
                        layer.params = [layer.w, layer.b]
                    self.params = [param for layer in self.layers for param in layer.params]

            # VALIDATE
            with torch.no_grad():
                for i in range(num_validation_batches):
                    X_val_mb = X_val[i * minibatch_size:(i + 1) * minibatch_size]
                    y_val_mb = y_val[i * minibatch_size:(i + 1) * minibatch_size]
                    activation = X_val_mb
                    for layer in self.layers:
                        if isinstance(layer, ConvPoolLayer):
                            layer.forward(activation, len(X_val_mb))
                        else:
                            layer.forward(activation, len(X_val_mb))
                        activation = layer.output
                    validation_accuracy += self.layers[-1].accuracy(torch.argmax(y_val_mb, dim=1)) * len(X_val_mb) / len(X_val)
                    validation_cost += self.layers[-1].cost(y_val_mb, 'validate') * len(X_val_mb) / len(X_val) + (l2_norm_squared/len(X_val)) * len(X_val_mb) / len(X_val)
                validation_costs.append(validation_cost.cpu())
                validation_scores.append(validation_accuracy.cpu())

                print(f'Epoch {epoch}: validation accuracy {validation_accuracy:.2%}')
                if validation_accuracy > best_validation_accuracy:
                    print("This is the best validation accuracy to date.")
                    best_validation_accuracy = validation_accuracy
                    best_iteration = iteration

        print("Finished training network.")
        print(f"Best validation accuracy {best_validation_accuracy:.2%} obtained at iteration {best_iteration}")

        return validation_costs, validation_scores #, test_score

class Network(BaseEstimator, TransformerMixin):
    def __init__(self,
                 sizes,
                 activation_functions,
                 learning_rate=0.1,
                 mini_batch_size=10,
                 cost="mean_squared_error",
                 optimizer = None,
                 init_method="naive",
                 weight_scale: float = 0.1,
                 bias_init = 0.001,
                 penalty=None,
                 alpha=0.1,
                 verbose=False,
                 random_state=None,
                 max_epochs = 50,
                 early_stop_patience = np.inf,
                 learning_schedule_patience = np.inf,
                 min_delta = 1e-4,
                 tol = 1e-3,
                 monitor_cost=False,
                 monitor_accuracy=False):
        self.sizes = sizes
        self.learning_rate = learning_rate
        self.activation_functions = activation_functions
        self.mini_batch_size = mini_batch_size
        self.optimizer = optimizer
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
        self.tol = tol
        self.monitor_cost = monitor_cost
        self.monitor_accuracy = monitor_accuracy
        self.random_state = random_state
        self.weights = None
        self.biases = None
        self.grads = None
        self.cost_value = None
        self.final_eta = None
        self.train_cost_cache = []
        self.val_cost_cache = []
        self.training_accuracy = []
        self.val_accuracy = []
        self.t = None
        self.n_epochs = None

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

            # Cost function
            cost_value = compute_cost(a_prev, y[:,n:n+self.mini_batch_size], self.cost)
            epoch_cost += cost_value * a_prev.shape[1]

            # Backward pass
            grads = []

            if len(caches_forward) < 2:
                a_prev = X[:, n:n + self.mini_batch_size]
            else:
                a_prev = caches_forward[-2]['a']

            if self.activation_functions[-1] == "sigmoid" and self.cost == 'cross_entropy':
                dz_L = caches_forward[-1]['a'] - y[:,n:n+self.mini_batch_size]
                dw_L = dz_L @ a_prev.T
                db_L = np.sum(dz_L, axis=1, keepdims=True)
                da_prev = caches_forward[-1]['weight'].T @ dz_L
                grads_L = {'dw': dw_L, 'db': db_L}
            else:
                da_L = cost_backward(caches_forward[-1]['a'], y[:,n:n+self.mini_batch_size], self.cost)
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

            # Penalties
            assert len(grads) == len(self.weights)
            if self.penalty == 'l2':
                self.weights = [
                    weight * (1-self.alpha*self.final_eta/X.shape[1])
                    for weight in self.weights
                ]
            elif self.penalty == 'l1':
                masks = [[weight > 0, weight < 0] for weight in self.weights]
                sign = [1, -1]
                self.weights = [
                    weight - self.final_eta * (self.alpha/X.shape[1]) * np.select(mask, sign, default=0)
                    for weight, mask in zip(self.weights, masks)
                ]

            # Weight update
            self.t += 1
            if self.optimizer is not None:
                self.weights, self.biases = self.optimizer.update_wb(self.weights,
                                                                     self.biases,
                                                                     grads,
                                                                     self.learning_rate,
                                                                     self.t)
            else:
                self.weights = [
                    weight - self.learning_rate * grad['dw']
                    for weight, grad in zip(self.weights, grads)
                ]
                self.biases = [
                    bias - self.learning_rate * grad['db']
                    for bias, grad in zip(self.biases, grads)
                ]
        epoch_cost /= X.shape[1]
        return epoch_cost

    def fit(self, X_train, y_train, X_val=None, y_val=None,):
        assert X_train.shape[1] == y_train.shape[1], \
            f"Mismatch in dim 1: n_samples in X_train is {X_train.shape[1]}, but y_train has {y_train.shape[1]}"

        assert X_train.shape[0] == self.sizes[0], \
            f"Features mismatch: n_features in X_train is {X_train.shape[0]}, but input layer expects {self.sizes[0]}"

        if X_val is not None:
            assert y_val is not None, "y_val must not be None"
            assert X_val.shape[1] == y_val.shape[1], \
                f"Mismatch in dim 1: n_samples in X_val is {X_val.shape[1]}, but y_val has {y_val.shape[1]}"

        if y_val is not None:
            assert X_val is not None, "X_val must not be None"
            assert X_val.shape[1] == y_val.shape[1], \
                f"Mismatch in dim 1: n_samples in X_val is {X_val.shape[1]}, but y_val has {y_val.shape[1]}"
            assert y_val.shape[0] == self.sizes[-1], \
                f"Target mismatch: Targets have {y_val.shape[0]} labels, but output layer expects {self.sizes[-1]}"

        assert y_train.shape[0] == self.sizes[-1], \
            f"Target mismatch: Targets have {y_train.shape[0]} labels, but output layer expects {self.sizes[-1]}"

        if self.random_state is None:
            self.random_state = 42

        self.t = 0

        # Initialize parameters
        self.weights, self.biases = map(list,
                                        zip(*Network.initialize_parameters(self.sizes,
                                                                           self.init_method,
                                                                           random_state=self.random_state,
                                                                           weight_scale=self.weight_scale,
                                                                           bias_init=self.bias_init)))
        # Momentum - Initialize Velocities
        if self.optimizer is not None:
            self.optimizer.initialize_vs(self.weights, self.biases)
        self.train_cost_cache = []
        self.val_cost_cache = []
        self.training_accuracy = []
        self.val_accuracy = []
        self.final_eta = self.learning_rate

        rng = np.random.default_rng(self.random_state)
        best_loss, val_loss, wait_sched, wait_stop, best_params = np.inf, np.nan, 0, 0, None
        self.n_epochs = 0
        for epoch in range(self.max_epochs):
            self.n_epochs += 1
            perm = rng.permutation(X_train.shape[1])
            X_shuffled = X_train[:,perm]
            y_shuffled = y_train[:,perm]
            epoch_cost = self.run_epoch(X_shuffled, y_shuffled)
            self.cost_value = epoch_cost

            if self.monitor_accuracy:
                train_score = self.score(X_train, y_train)
                val_score = self.score(X_val, y_val) if X_val is not None else None

                self.training_accuracy.append(train_score)
                self.val_accuracy.append(val_score)

            if self.verbose:
                print(f'epoch: {epoch}, cost: {epoch_cost}')
                if self.monitor_accuracy:
                    print(f'train score: {train_score}, val score: {val_score}\n ')

            if self.monitor_cost:
                if X_val is not None:
                    a_val = self.predict_proba(X_val)
                    val_loss = compute_cost(a_val, y_val, self.cost)
                    self.val_cost_cache.append(val_loss)
                self.train_cost_cache.append(epoch_cost)

            # Early Stop
            if (self.early_stop_patience < np.iinfo(np.int64).max) or (self.learning_schedule_patience < np.iinfo(np.int64).max):
                assert X_val is not None, "No validation data passed"
                a_val = self.predict_proba(X_val)
                val_loss = compute_cost(a_val, y_val, self.cost)

                if val_loss < best_loss - self.min_delta:
                    best_loss, wait_sched, wait_stop = val_loss, 0, 0
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
        assert X.shape[0] == self.sizes[0], \
            f"Features mismatch: n_features in X_train is {X.shape[0]}, but input layer expects {self.sizes[0]}"

        a_prev = X
        for l in range(len(self.weights)):
            cache = Network.feedforward(a_prev, self.weights[l], self.biases[l],
                                        activation=self.activation_functions[l])
            a_prev = cache['a']

        return a_prev

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=0)

    def score(self, X, y):
        ypred = self.predict(X)
        y = np.argmax(y, axis=0)
        return accuracy_score(y, ypred)

    def save(self, filename):
        data = {'sizes': self.sizes,
                'weights': [w.tolist() for w in self.weights],
                'biases': [b.tolist() for b in self.biases],
                'activation_functions': self.activation_functions,
                'cost': self.cost,}
        with open(filename, 'w') as f:
            json.dump(data, f)




