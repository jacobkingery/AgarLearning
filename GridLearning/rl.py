import numpy as np
import random

class ReinforcementQLearning:
	def __init__(self, neuralNet, numberOfActions, exploratoryRateFn, backpropLength, discountFactorFn, learningRateFn, randomSeed=random.random()):
		# Learning rate of 1 is optimal for deterministic environments
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

	def getAction(self, state, i=0, evaluation=False):
		exploratoryRate = self.exploratoryRateFn(i)
		if (not evaluation) and (random.random() < exploratoryRate):
			return random.choice(self.actions), True
		else:
			return self.getBestActionValuePair(state)[0], False

	def getBestActionValuePair(self, state):
		actionValuePairs = self.neuralNet.getValues(state)
		bestAction = np.argmax(actionValuePairs)
		return (bestAction, actionValuePairs[bestAction])

	def storeSARS(self, state, action, reward, newState):
		# stateBestActionValuePair = self.getBestActionValuePair(state)
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
		learningRate = self.learningRateFn(i)
		discountFactor = self.discountFactorFn(i)
		trainingTuples = []
		while self.currentSAVRSAVIndex < len(self.SAVRSAV):
			currentSAVRSAV = self.SAVRSAV[self.currentSAVRSAVIndex]
			value = currentSAVRSAV['predictedValueOfAction'] + learningRate * (currentSAVRSAV['reward'] + discountFactor*currentSAVRSAV['predictedValueOfNewAction'] - currentSAVRSAV['predictedValueOfAction'])
			trainingTuples.append((currentSAVRSAV['state'],currentSAVRSAV['action'],value))
			self.currentSAVRSAVIndex += 1
		return self.neuralNet.train(trainingTuples)

	def backprop(self):
		pass

if __name__ == "__main__":
	class DumbNeuralNet:
		def getValues(self, state):
			return [random.random() for _ in range(4)]

		def train(self, trainingTuples):
			# print trainingTuples
			return True

	exampleState1 = [0,0,0,0]
	exampleState2 = [1,1,1,1]
	neuralNet = DumbNeuralNet()
	numberOfActions = 4
	backpropLength = 0
	
	def expRate(i):
		return 0.3
	def discountFactor(i):
		return 0.5
	def learningRate(i):
		return 1

	qRL = ReinforcementQLearning(neuralNet, numberOfActions, expRate, backpropLength, discountFactor, learningRate)
	i = 0
	print qRL.getAction(exampleState1, i)
	print qRL.storeSARS(exampleState1, 1, 2, exampleState2)
	print qRL.storeSARS(exampleState2, 3, 0, exampleState1)
	print qRL.train(i)
	
