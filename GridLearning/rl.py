import numpy as np
import random

class ReinforcementQLearning:
	"""
    Class for a reinforcement q learning
    """
	def __init__(self, neuralNet, numberOfActions, exploratoryRateFn, backpropLength, discountFactorFn, learningRateFn, randomSeed=random.random()):
		"""
        Initializes Class
        Inputs:
            neuralNet  		  - A nueral net (as defined in nn.py)
            numberOfActions   - number of possible actions
            exploratoryRateFn - function that determines the exploratory rate
            backpropLength    - currently unsed, sets how far back rewards propogate
            discountFactorFn  - function that determines the discount factor
            learningRateFn	  - function that determines the learning rate
            randomSeed		  - random seed
        """
		random.seed(randomSeed)
		self.neuralNet = neuralNet
		self.numberOfActions = numberOfActions
		self.exploratoryRateFn = exploratoryRateFn
		self.backpropLength = backpropLength
		self.discountFactorFn = discountFactorFn
		self.learningRateFn = learningRateFn
		self.actions = range(numberOfActions)
		self.SAVRSAV = []
		self.currentSAVRSAVIndex = 0

	def getAction(self, state, i=0, evaluation=False, useB = True):
		"""
        Returns an action given the state of the game
        Inputs:
            state - current game state
            i - iterations completed
            evaluation - whether to always return the best move
            useB - whether to use boltzman's distribution for selecting exploratory moves
        """
		exploratoryRate = self.exploratoryRateFn(i)
		
		
		if (not evaluation and useB):
			actionValuePairs = self.neuralNet.getValues(state)

			boltzmann = [np.exp(value/exploratoryRate) for value in actionValuePairs]
			boltzmannSum = sum(boltzmann)

			boltzmann = [prob/boltzmannSum for prob in boltzmann]

			cutOffValues = np.cumsum(boltzmann)
			randomNumber = random.random()
			action = np.argmax(cutOffValues > randomNumber)
			
			isExploratory = (action == np.argmax(boltzmann))
			
			# print action, isExploratory
			return action, isExploratory
			
		elif (not evaluation and (random.random() < exploratoryRate)):
			return random.choice(range(4)), True

		else:
			return self.getBestActionValuePair(state)[0], False

	def getBestActionValuePair(self, state):
		"""
        Returns the best action and it's value (as determined by the nueral net)
        Inputs:
            state - current game state
        """
		actionValuePairs = self.neuralNet.getValues(state)
		bestAction = np.argmax(actionValuePairs)
		return (bestAction, actionValuePairs[bestAction])

	def storeSARS(self, state, action, reward, newState):
		"""
        Stores the state, action, reward, newState tuple
        Inputs:
            state - previous game state
            action - the action taken
            reward - the reward recieved
            newState - new game state
        """
		newStateBestActionValuePair = self.getBestActionValuePair(newState)
		SAVRSAVdict = {'state': state,
					   'action': action,
					   'predictedValueOfAction': self.neuralNet.getValues(state)[action], 
					   'reward': reward,
					   'newState': newState,
					   'newBestAction': newStateBestActionValuePair[0],
					   'predictedValueOfNewAction':newStateBestActionValuePair[1]}
		self.SAVRSAV.append(SAVRSAVdict)
		return SAVRSAVdict

	def train(self, i):
		"""
        Trains the neural network on all new stored state, action, reward, newState tuples
        Inputs:
            i - iterations completed
        """
		learningRate = self.learningRateFn(i)
		discountFactor = self.discountFactorFn(i)
		trainingTuples = []
		while self.currentSAVRSAVIndex < len(self.SAVRSAV):
			currentSAVRSAV = self.SAVRSAV[self.currentSAVRSAVIndex]
			value = currentSAVRSAV['predictedValueOfAction'] + learningRate * (currentSAVRSAV['reward'] + discountFactor*currentSAVRSAV['predictedValueOfNewAction'] - currentSAVRSAV['predictedValueOfAction'])
			trainingTuples.append((currentSAVRSAV['state'],currentSAVRSAV['action'],value))
			self.currentSAVRSAVIndex += 1
		return self.neuralNet.train(trainingTuples)