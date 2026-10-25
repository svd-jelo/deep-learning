import json
from src.network import Network
import numpy as np

def load_network(filename):
    with open(filename) as f:
        data = json.load(f)
    net = Network(sizes=data['sizes'],
                  activation_functions=data['activation_functions'],
                  cost=data['cost'])
    net.weights = [np.array(w) for w in data['weights']]
    net.biases = [np.array(b) for b in data['biases']]
    return net
