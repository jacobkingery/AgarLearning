import numpy as np
import random

class ReinforcementQLearning:
	def __init__(self, neuralNet, numberOfActions, exploratoryRate, backpropLength, discountFactor, learningRate=1, randomSeed=random.random()):
		# Learning rate of 1 is optimal for deterministic environments
		random.seed(randomSeed)
		self.neuralNet = neuralNet
		self.numberOfActions = numberOfActions
		self.exploratoryRate = exploratoryRate
		self.backpropLength = backpropLength
		self.discountFactor = discountFactor
		self.learningRate = learningRate
		self.actions = range(numberOfActions)
		self.SAVRSAV = [] 
		self.currentSAVRSAVIndex = 0

	def getAction(self, state):
		if random.random()<self.exploratoryRate:
			return random.choice(self.actions)
		else:
			return self.getBestActionValuePair(state)[0]

	def getBestActionValuePair(self, state):
<<<<<<< HEAD
		actionValuePairs = self.neuralNet.getValuesForActions(state, self.actions) #expects a list of tuples: [(action, value),(action, value),(action, value)]
		bestActionValuePair = max(actionValuePairs, key=lambda pair: pair[1])
		return bestActionValuePair
=======
		actionValuePairs = self.neuralNet.getValues(state)
		bestAction = np.argmax(actionValuePairs)
		return (bestAction, actionValuePairs[bestAction])
>>>>>>> integration

	def storeSARS(self, state, action, reward, newState):
		stateBestActionValuePair = self.getBestActionValuePair(state)
		newStateBestActionValuePair = self.getBestActionValuePair(newState)
		SAVRSAVdict = {'state': state,
					   'action': action,
					   'predictedValueOfAction': stateBestActionValuePair[1], 
					   'reward': reward,
					   'newState': newState,
					   'newBestAction': newStateBestActionValuePair[0],
					   'predictedValueOfNewAction':newStateBestActionValuePair[1]}
		self.SAVRSAV.append(SAVRSAVdict)
		return SAVRSAVdict

	def train(self):
		trainingTuples = []
		while self.currentSAVRSAVIndex < len(self.SAVRSAV):
			currentSAVRSAV = self.SAVRSAV[self.currentSAVRSAVIndex]
			value = currentSAVRSAV['predictedValueOfAction'] + self.learningRate * (currentSAVRSAV['reward'] + self.discountFactor*currentSAVRSAV['predictedValueOfNewAction'] - currentSAVRSAV['predictedValueOfAction'] )
			trainingTuples.append((currentSAVRSAV['state'],currentSAVRSAV['action'],value))
			self.currentSAVRSAVIndex += 1
		return self.neuralNet.train(trainingTuples)

	def backprop(self):
		pass

<<<<<<< HEAD
if __name__ == "__main__":
	class DumbNeuralNet:
		def getValuesForActions(self, state, actions):
			actionValueList = []
			for action in actions:
				actionValueList.append((action, random.random()))
			return actionValueList
=======
class DumbNeuralNet:
	def getValues(self, state, actions):
		actionValueList = []
		for action in actions:
			actionValueList.append((action, random.random()))
		return actionValueList
>>>>>>> integration

	def train(self, trainingTuples):
		# print trainingTuples
		return True

if __name__ == "__main__":
	exampleState1 = [ 0 , 0, 0, 0]
	exampleState2 = [ 1,1,1,1]
	neuralNet = DumbNeuralNet()
	numberOfActions = 4
	exploratoryRate = 0
	backpropLength = 0
	discountFactor = .5
	qRL = ReinforcementQLearning(neuralNet, numberOfActions, exploratoryRate, backpropLength, discountFactor)
	print qRL.getAction(exampleState1)
	print qRL.storeSARS(exampleState1, 1, 2, exampleState2)
	print qRL.storeSARS(exampleState2, 3, 0, exampleState1)
	print qRL.train()
	
