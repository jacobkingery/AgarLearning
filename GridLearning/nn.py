import tensorflow as tf
import numpy as np

# Some of the following code was adapted from
# https://github.com/nivwusquorum/tensorflow-deepq

class Layer(object):
    """
    Class for a NN layer
    """
    def __init__(self, numInputs, actFunc, numOutputs, seed=None, scope='layer'):
        """
        Initializes layer
        Inputs:
            numInputs  - number of inputs to this layer
            actFunc    - activation function between this layer and the next
            numOutputs - number of outputs from this layer
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

        self.layers = [None] * (len(layers) - 1)
        seeds = seeds or [None] * len(self.layers)
        with tf.variable_scope(scope):
            for i in range(len(self.layers)):
                self.layers[i] = Layer(
                    layers[i][0],
                    layers[i][1],
                    layers[i+1][0],
                    seeds[i],
                    'layer{}'.format(i)
                )

    def __call__(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return self.outputFunc(y)

class NeuralNet(object):
    """
    Overall class containing our NN
    """
    def __init__(self, layers, optimizer=None, seeds=None):
        """
        Initializes NN container
        Inputs:
            layers    - list of tuples containing the number of nodes and
                        activation function for each layer; see MLP for example
            optimizer - optimizer for prediction error; default
                        tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9)
            seeds     - list of integer seeds for layer random initializations,
                        must be of length len(layers)-1; default [0,1,...,N]
        """
        optimizer = optimizer or tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9)
        seeds = seeds or range(len(layers) - 1)
        self.neuralNet = MLP(layers, seeds)

        # self.session = tf.InteractiveSession()
        self.session = tf.Session()

        self.stateSize = layers[0][0]
        self.numActions = layers[-1][0]

        self.observation  = tf.placeholder(tf.float32, (None, self.stateSize))
        self.actionScores = self.neuralNet(self.observation)

        self.actionMask = tf.placeholder(tf.float32, (None, self.numActions))
        self.values = tf.placeholder(tf.float32, (None,))
        self.maskedActionScores = tf.reduce_sum(self.actionScores * self.actionMask, reduction_indices=[1,])
        self.predictionError = tf.reduce_mean(tf.square(self.maskedActionScores - self.values))
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
        states = np.empty((len(SAVs), self.stateSize))
        actionMask = np.zeros((len(SAVs), self.numActions))
        values = np.empty((len(SAVs),))

        for i, (state, action, value) in enumerate(SAVs):
            states[i] = state
            actionMask[i][action] = 1
            values[i] = value

        self.session.run([
            self.predictionError,
            self.trainOp
        ], {
            self.observation: states,
            self.actionMask: actionMask,
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
        return self.session.run(self.actionScores, {
            self.observation: state
        })[0]


if __name__ == '__main__':
    layers = [
        (9,tf.tanh),    # input
        (8,tf.tanh),    # hiddens
        (6,tf.tanh),
        (4,tf.identity) # output
    ]
    nn = NeuralNet(layers)

    state = np.array([
        [2,1,0],
        [0,1,0],
        [0,0,0]
    ]).flatten()[np.newaxis,:]
    print nn.getValues(state)

    action = 1
    reward = 10
    newState = np.array([
        [0,2,0],
        [0,1,0],
        [0,0,0]
    ]).flatten()[np.newaxis,:]
    nn.train([(state, action, reward)])

    state = newState
    print nn.getValues(state)

    action = 0
    reward = -10
    newState = np.array([
        [0,2,0],
        [0,1,0],
        [0,0,0]
    ]).flatten()[np.newaxis,:]
    nn.train([(state, action, reward)])

    state = newState
    print nn.getValues(state)

    action = 0
    reward = -10
    newState = np.array([
        [0,2,0],
        [0,1,0],
        [0,0,0]
    ]).flatten()[np.newaxis,:]
    nn.train([(state, action, reward)])

    state = newState
    print nn.getValues(state)
