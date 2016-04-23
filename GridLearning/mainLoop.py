import game
import rl
import nn
import random

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tqdm

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
discountFactor = 0.9

stateSize = gameX * gameY
inputActFunc = tf.tanh
hiddenLayers = [(25, tf.tanh)]
mode = 0

myNN = nn.NeuralNet(stateSize, numActions, inputActFunc, hiddenLayers, mode=mode)
myRl = rl.ReinforcementQLearning(myNN, numActions, expRate, bpLength, discountFactor, learningRate=1, randomSeed=randomSeed)

numGames = 1000
numMovesTaken = []
numGamesEvalIterationList = []
numGamesEvalAverageList = []
print('Playing {0} games'.format(str(numGames)))
for i in tqdm.tqdm(range(numGames)):
	myGame = game.Game(gameX,gameY,numFood,i)
	# if i == 0:
	# 	myGame.printGameState()

	while (not myGame.isGameOver()):
		currentState = myGame.flattenGameState()
		action = myRl.getAction(currentState)
		reward = myGame.updateGameState(action)
		nextState = myGame.flattenGameState()

		myRl.storeSARS(currentState, action, reward, nextState)
		myRl.train()

		# if i == numGames-1:
		# 	myGame.printGameState()

	numMovesTaken.append(myGame.numMoves)

	if i%50 ==0:
		numGamesEvalList = []
		for j in range(20):
			myGameEval = game.Game(gameX,gameY,numFood,j*.11)

			while (not myGame.isGameOver()):
				currentState = myGameEval.flattenGameState()
				action = myRl.getAction(currentState, evaluation=False)
				reward = myGameEval.updateGameState(action)
				nextState = myGameEval.flattenGameState()

			numGamesEvalList.append(myGame.numMoves)
		numGamesEvalIterationList.append(i)
		numGamesEvalAverageList.append(np.mean(numGamesEvalList))


plt.bar(numGamesEvalIterationList, numGamesEvalAverageList)
plt.xlabel('games trained on')
plt.ylabel('eval average number of moves')
plt.title('Different Game Every Time: mode ' + str(mode))
plt.show()

	

includeInAvg = 10
runningAverage = runningMean(numMovesTaken, includeInAvg)

plt.plot(numMovesTaken)
plt.plot(runningAverage, 'r')
plt.legend(['number of moves taken', 'sliding average ({})'.format(includeInAvg)])
plt.xlabel('game number')
plt.ylabel('number of moves')
plt.title('Different Game Every Time: mode ' + str(mode))
plt.show()
