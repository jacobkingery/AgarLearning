import tensorflow as tf
import numpy as np

# Some of the following code was adapted from
# https://github.com/nivwusquorum/tensorflow-deepq

class Layer(object):
    """
    Class for a NN layer
    """
    def __init__(self, numInputs, numOutputs, actFunc, seed=None, scope='layer'):
        """
        Initializes layer
        Inputs:
            numInputs  - number of inputs to this layer
            numOutputs - number of outputs from this layer
            actFunc    - activation function between this layer and the next
            seed       - integer seed for random initialization; default None
            scope      - TF variable scope; must be unique for each layer;
                         default 'layer'
        """
        self.scope = scope
        self.func = actFunc
        with tf.variable_scope(self.scope):
            self.W = tf.get_variable('W', (numInputs, numOutputs),
                initializer=tf.random_uniform_initializer(
                    -1.0 / np.sqrt(numInputs),
                    1.0 / np.sqrt(numInputs),
                    seed=seed
                )
            )
            self.b = tf.get_variable('b', (numOutputs,), initializer=tf.constant_initializer(0))

    def __call__(self, x):
        with tf.variable_scope(self.scope):
            return self.func(tf.matmul(x, self.W) + self.b)

class MLP(object):
    """
    Class for a NN (a.k.a. MultiLayer Perceptron)
    """
    def __init__(self, layers, seeds=None, scope='mlp'):
        """
        Initializes NN
        Inputs:
            layers - list of tuples containing the number of nodes and activation
                     function for each layer; e.g.
                     [(10, tf.tanh), (6, tf.tanh), (4, tf.identity)]
                     for 10 inputs, a hidden layer with 6 nodes, and 4 outputs
            seeds  - list of integer seeds for layer random initializations,
                     must be of length len(layers)-1; default None
            scope  - TF variable scope; must be unique for each layer; default 'mlp'
        """
        self.outputFunc = layers[-1][1]

        self.layers = []
        seeds = seeds or [None] * (len(layers) - 1)
        with tf.variable_scope(scope):
            for i in range(1, len(layers)):
                self.layers.append(Layer(
                    layers[i-1][0],
                    layers[i][0],
                    layers[i][1],
                    seeds[i-1],
                    'layer{}'.format(i)
                ))

    def __call__(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return self.outputFunc(y)

class NeuralNet(object):
    """
    Overall class containing our NN
    """
    def __init__(self, stateSize, numActions, hiddenLayers, outputActFunc=tf.identity, optimizer=None, seeds=None, mode=0):
        """
        Initializes NN container
        Inputs:
            stateSize     - size of state array
            numActions    - number of actions
            hiddenLayers  - list of tuples containing the number of nodes and
                            activation function for each hidden layer; e.g.
                            [(10, tf.tanh), (5, tf.tanh)]
                            for two hidden layers of size 10 and 5
            outputActFunc - activation function after the output layer;
                            default tf.identity
            optimizer     - optimizer for prediction error; default
                            tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9)
            seeds         - list of integer seeds for layer random initializations,
                            must be of length len(layers)-1; default [0,1,...,N]
            mode          - NN input/output mode; default 0
                                0 for state in, all values out
                                1 for state and action in, one value out
        """
        optimizer = optimizer or tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9)
        self.stateSize = stateSize
        self.numActions = numActions
        self.mode = mode

        if self.mode == 0:
            self.inputSize = self.stateSize
            self.outputSize = self.numActions
        elif self.mode == 1:
            self.inputSize = self.stateSize + self.numActions
            self.outputSize = 1
        else:
            raise NotImplementedError('Mode {} does not exist'.format(self.mode))

        layers = [(self.inputSize, None)] + hiddenLayers + [(self.outputSize, outputActFunc)]
        seeds = seeds or range(len(layers) - 1)
        self.neuralNet = MLP(layers, seeds)

        # self.session = tf.InteractiveSession()
        self.session = tf.Session()


        self.inputNN = tf.placeholder(tf.float32, (None, self.inputSize))
        self.outputNN = self.neuralNet(self.inputNN)

        self.outputMask = tf.placeholder(tf.float32, (None, self.outputSize))
        self.values = tf.placeholder(tf.float32, (None,))
        self.maskedOutput = tf.reduce_sum(self.outputNN * self.outputMask, reduction_indices=[1,])
        self.predictionError = tf.reduce_mean(tf.square(self.maskedOutput - self.values))
        gradients = optimizer.compute_gradients(self.predictionError)
        # for i, (grad, var) in enumerate(gradients):
        #     if grad is not None:
        #         gradients[i] = (tf.clip_by_norm(grad, 5), var)
        self.trainOp = optimizer.apply_gradients(gradients)

        self.session.run(tf.initialize_all_variables())

    def train(self, SAVs):
        """
        Updates NN with given information
        Inputs:
            SAVs - list of tuples of (state, action, value)
        """
        states = np.empty((len(SAVs), self.inputSize))
        outputMask = np.zeros((len(SAVs), self.outputSize))
        values = np.empty((len(SAVs),))

        if self.mode == 0:
            for i, (state, action, value) in enumerate(SAVs):
                states[i] = state
                outputMask[i][action] = 1
                values[i] = value

        elif self.mode == 1:
            outputMask[:,:] = 1
            actions = np.identity(self.numActions)[:,np.newaxis]

            for i, (state, action, value) in enumerate(SAVs):
                states[i] = np.concatenate((state, actions[action]), axis=1)
                values[i] = value

        self.session.run([
            self.predictionError,
            self.trainOp
        ], {
            self.inputNN: states,
            self.outputMask: outputMask,
            self.values: values
        })

    def getValues(self, state):
        """
        Inputs state into NN and returns resulting action scores
        Inputs:
            state - flattened numpy array representing current state
        Outputs:
            numpy array of scores
        """
        if self.mode == 0:
            scores = self.session.run(self.outputNN, {
                self.inputNN: state
            })[0]
        elif self.mode == 1:
            actions = np.identity(self.numActions)[:,np.newaxis]
            scores = np.zeros((self.numActions,))
            for i,action in enumerate(actions):
                scores[i] = self.session.run(self.outputNN, {
                    self.inputNN: np.concatenate((state, action), axis=1)
                })[0][0]
        return scores

    def getValuesForActions(self, state, actions):
        """
        Inputs state and actions into NN and returns resulting value for each action
        Inputs:
            state - flattened numpy array representing current state
            actions - list of actions
        Outputs:
            numpy array of scores for each action
        """
        actionValuePairs = []
        for i, action in enumerate(actions):
            value = self.getValues(state + [action])
            actionValuePairs.append((action,value))
        return actionValuePairs


if __name__ == '__main__':
    stateSize = 9
    numActions = 4
    hiddenLayers = [
        (25,tf.tanh),    # hiddens
    ]
    nn = NeuralNet(stateSize, numActions, hiddenLayers)

    state = np.array([
        [2,0,0],
        [1,1,0],
        [0,0,0]
    ]).flatten()[np.newaxis,:]
    print nn.getValues(state)

    action = 2
    reward = 1
    newState = np.array([
        [0,0,0],
        [2,1,0],
        [0,0,0]
    ]).flatten()[np.newaxis,:]
    nn.train([(state, action, reward)])

    state = newState
    print nn.getValues(state)

    action = 2
    reward = 0
    newState = np.array([
        [0,0,0],
        [0,1,0],
        [2,0,0]
    ]).flatten()[np.newaxis,:]
    nn.train([(state, action, reward)])

    state = newState
    print nn.getValues(state)

    action = 2
    reward = -1
    newState = np.array([
        [0,0,0],
        [0,1,0],
        [2,0,0]
    ]).flatten()[np.newaxis,:]
    nn.train([(state, action, reward)])

    state = newState
    print nn.getValues(state)
