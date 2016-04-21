import game
import rl
import nn

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def runningMean(arr, numPoints):
	runningAvg = []
	for i,val in enumerate(arr):
		
		if i < numPoints:
			avg = float(sum(arr[0:i+1]))/(i+1)
		else:
			avg = float(sum(arr[i-numPoints+1:i+1]))/(numPoints)

		runningAvg.append(avg)

	return runningAvg

randomSeed = 42
gameX = 3
gameY = 3
numFood = 3
numActions = 4
expRate = 0.05
bpLength = 10
discountFactor = 0.5

stateSize = gameX * gameY
inputActFunc = tf.tanh
hiddenLayers = [(25, tf.tanh)]
mode = 1

myNN = nn.NeuralNet(stateSize, numActions, inputActFunc, hiddenLayers, mode=mode)
myRl = rl.ReinforcementQLearning(myNN, numActions, expRate, bpLength, discountFactor, learningRate=1, randomSeed=randomSeed)

numGames = 100
numMovesTaken = []
for i in range(numGames):
	myGame = game.Game(gameX,gameY,numFood,1)

	while (not myGame.isGameOver()):
		currentState = myGame.flattenGameState()
		action = myRl.getAction(currentState)
		reward = myGame.updateGameState(action)
		nextState = myGame.flattenGameState()

		myRl.storeSARS(currentState, action, reward, nextState)
		myRl.train()

		# if i == numGames-1:
			# myGame.printGameState()

	numMovesTaken.append(myGame.numMoves)

includeInAvg = 100
runningAverage = runningMean(numMovesTaken, includeInAvg)

plt.plot(numMovesTaken)
plt.plot(runningAverage, 'r')
plt.legend(['number of moves taken', 'sliding average ({})'.format(includeInAvg)])
plt.xlabel('game number')
plt.ylabel('number of moves')
plt.title('mode {}'.format(mode))
plt.show()
