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

    def variables(self):
        return [self.b,  self.W]

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

    def variables(self):
        res = []
        for layer in self.layers:
            res.extend(layer.variables())
        return res

class NeuralNet(object):
    """
    Overall class containing our NN
    """
    def __init__(self, layers, optimizer, seeds=None, discountRate=0.95, targetNetUpdateRate=0.01):
        """
        Initializes NN container
        Inputs:
            layers       - list of tuples containing the number of nodes and
                           activation function for each layer; see MLP for example
            optimizer    - optimizer for prediction error; e.g.
                           tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9)
            seeds        - list of integer seeds for layer random initializations,
                           must be of length len(layers)-1; default [0,1,...,N]
            discountRate - discount rate for predicting future rewards;
                           default 0.95
            targetNetUpdateRate - update rate for target net; default 0.01
        """
        seeds = seeds or range(len(layers) - 1)
        self.neuralNet = MLP(layers, seeds, 'neuralNet')
        self.targetNet = MLP(layers, seeds, 'targetNet')

        discountRate = tf.constant(discountRate)
        targetNetUpdateRate = tf.constant(targetNetUpdateRate)

        # self.session = tf.InteractiveSession()
        self.session = tf.Session()

        self.stateSize = layers[0][0]
        self.numActions = layers[-1][0]

        # for action predictions
        with tf.name_scope('action'):
            self.observation  = tf.placeholder(tf.float32, (None, self.stateSize))
            self.actionScores = self.neuralNet(self.observation)

        # for target future reward predictions
        with tf.name_scope('target'):
            self.nextObservation  = tf.placeholder(tf.float32, (None, self.stateSize))
            self.nextObservationMask = tf.placeholder(tf.float32, (None,))
            self.nextActionScores = tf.stop_gradient(self.targetNet(self.nextObservation))
            self.rewards  = tf.placeholder(tf.float32, (None,))
            targetValues = tf.reduce_max(self.nextActionScores, reduction_indices=[1,]) * self.nextObservationMask
            self.futureRewards = self.rewards + discountRate * targetValues

        # for prediction errors
        with tf.name_scope('error'):
            self.actionMask = tf.placeholder(tf.float32, (None, self.numActions))
            self.maskedActionScores = tf.reduce_sum(self.actionScores * self.actionMask, reduction_indices=[1,])
            self.predictionError = tf.reduce_mean(tf.square(self.maskedActionScores - self.futureRewards))
            gradients = optimizer.compute_gradients(self.predictionError)
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, 5), var)
            self.trainOp = optimizer.apply_gradients(gradients)

        # for target net updates
        with tf.name_scope('update'):
            self.targetNetUpdate = []
            for vSource, vTarget in zip(self.neuralNet.variables(), self.targetNet.variables()):
                # this is equivalent to target = (1-alpha) * target + alpha * source
                self.targetNetUpdate.append(
                    vTarget.assign_sub(targetNetUpdateRate * (vTarget - vSource))
                )
            self.targetNetUpdate = tf.group(*self.targetNetUpdate)

        self.session.run(tf.initialize_all_variables())
        self.session.run(self.targetNetUpdate)

    def train(self, SARS):
        """
        Updates NN with given information
        Inputs:
            SARS - list of tuples of (state, action, reward, newState)
        """
        states = np.empty((len(SARS), self.stateSize))
        actionMask = np.zeros((len(SARS), self.numActions))
        rewards = np.empty((len(SARS),))
        newStates = np.zeros((len(SARS), self.stateSize))
        newStatesMask = np.zeros((len(SARS),))

        for i, (state, action, reward, newState) in enumerate(SARS):
            states[i] = state
            actionMask[i][action] = 1
            rewards[i] = reward
            if newState is not None:
                newStates[i] = newState
                newStatesMask[i] = 1

        self.session.run([
            self.predictionError,
            self.trainOp
        ], {
            self.observation: states,
            self.nextObservation: newStates,
            self.nextObservationMask: newStatesMask,
            self.actionMask: actionMask,
            self.rewards: rewards
        })

        self.session.run(self.targetNetUpdate)

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
    optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.9)
    nn = NeuralNet(layers, optimizer)

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
    nn.train([(state, action, reward, newState)])

    state = newState
    print nn.getValues(state)

    action = 0
    reward = -10
    newState = np.array([
        [0,2,0],
        [0,1,0],
        [0,0,0]
    ]).flatten()[np.newaxis,:]
    nn.train([(state, action, reward, newState)])

    state = newState
    print nn.getValues(state)

    action = 0
    reward = -10
    newState = np.array([
        [0,2,0],
        [0,1,0],
        [0,0,0]
    ]).flatten()[np.newaxis,:]
    nn.train([(state, action, reward, newState)])

    state = newState
    print nn.getValues(state)
