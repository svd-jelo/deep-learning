import numpy as np

class MomentumOptimizer:
    def __init__(self, beta):
        self.beta = beta
        self.v_weights = []
        self.v_biases = []

    def initialize_vs(self, weights, biases):
        assert len(weights) == len(biases), \
            f"Size mismatch: weights have size {len(weights)}, but biases have size {len(biases)}"
        self.v_weights = [self.initialize_param(weight.shape) for weight in weights]
        self.v_biases = [self.initialize_param(bias.shape) for bias in biases]

    @staticmethod
    def initialize_param(shape):
        return np.zeros(shape)

    def update_wb(self, weights, biases, grads, learning_rate, t=None):
        # velocity update
        self.v_weights = [
            self.beta * v_weight + (1-self.beta) * grad['dw']
            for v_weight, grad in zip(self.v_weights, grads)
        ]
        self.v_biases = [
            self.beta * v_bias + (1-self.beta) * grad['db']
            for v_bias, grad in zip(self.v_biases, grads)
        ]
        # weight update
        weight_update = [
            weight - learning_rate * v_weight
            for weight, v_weight in zip(weights, self.v_weights)
        ]
        bias_update = [
            bias - learning_rate * v_bias
            for bias, v_bias in zip(biases, self.v_biases)
        ]
        return weight_update, bias_update

class AdamOptimizer:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.v_weights = []
        self.v_biases = []
        self.s_weights = []
        self.s_biases = []

    def initialize_vs(self, weights, biases):
        assert len(weights) == len(biases), \
            f"Size mismatch: weights have size {len(weights)}, but biases have size {len(biases)}"
        self.v_weights = [np.zeros(weight.shape) for weight in weights]
        self.v_biases = [np.zeros(bias.shape) for bias in biases]
        self.s_weights = [np.zeros(weight.shape) for weight in weights]
        self.s_biases = [np.zeros(bias.shape) for bias in biases]

    def update_wb(self, weights, biases, grads, learning_rate, t):
        #v update
        self.v_weights = [
            self.beta1 * v_weight + (1-self.beta1) * grad['dw']
            for v_weight, grad in zip(self.v_weights, grads)
        ]
        self.v_biases = [
            self.beta1 * v_bias + (1-self.beta1) * grad['db']
            for v_bias, grad in zip(self.v_biases, grads)
        ]
        self.s_weights = [
            self.beta2 * s_weight + (1-self.beta2) * grad['dw']**2
            for s_weight, grad in zip(self.s_weights, grads)
        ]
        self.s_biases = [
            self.beta2 * s_bias + (1-self.beta2) * grad['db']**2
            for s_bias, grad in zip(self.s_biases, grads)
        ]

        corr_fact1 = 1 / (1-self.beta1**t)
        corr_fact2 = 1 / (1-self.beta2**t)
        weight_update = [
            weight - learning_rate * corr_fact1 * v_weight / (np.sqrt(corr_fact2 * s_weight) + self.epsilon)
            for weight, v_weight, s_weight in zip(weights, self.v_weights, self.s_weights)
        ]
        bias_update = [
            bias - learning_rate * corr_fact1 * v_bias / (np.sqrt(corr_fact2 * s_bias) + self.epsilon)
            for bias, v_bias, s_bias in zip(biases, self.v_biases, self.s_biases)
        ]
        return weight_update, bias_update