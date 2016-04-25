import game
import rl
import nn
import random
import vis

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
numFood = 1
numActions = 4
expRate = 0.2
bpLength = 10
discountFactor = 0.5

stateSize = gameX * gameY
hiddenLayers = [(25, tf.tanh)]
mode = 0

myNN = nn.NeuralNet(stateSize, numActions, hiddenLayers, mode=mode)
myRl = rl.ReinforcementQLearning(myNN, numActions, expRate, bpLength, discountFactor, learningRate=1, randomSeed=randomSeed)

numGames = 1000
numMovesTaken = []
numGamesEvalIterationList = []
numGamesEvalAverageList = []
numGamesEvalMedianList = []
print('Playing {0} games'.format(str(numGames)))
for i in tqdm.tqdm(range(numGames)):
	myGame = game.Game(gameX,gameY,numFood,i)

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

	if i%50 == 0 and i!=0:
		numGamesEvalList = []
		for j in range(20):
			myGameEval = game.Game(gameX,gameY,numFood,j*.11)

			while (not myGameEval.isGameOver()) and myGameEval.numMoves < 100:
				currentState = myGameEval.flattenGameState()
				action = myRl.getAction(currentState, evaluation=True)
				reward = myGameEval.updateGameState(action)
				nextState = myGameEval.flattenGameState()
			print myGameEval.numMoves
			numGamesEvalList.append(myGameEval.numMoves)
		numGamesEvalIterationList.append(i)
		numGamesEvalAverageList.append(np.mean(numGamesEvalList))
		numGamesEvalMedianList.append(np.median(numGamesEvalList))



# plt.bar(np.array(numGamesEvalIterationList)-5, numGamesEvalAverageList,color='b',width=10)
# plt.bar(np.array(numGamesEvalIterationList)+5, numGamesEvalMedianList,color='g',width=10)
# plt.legend(['mean','median'])
# plt.xlabel('games trained on')
# plt.ylabel('eval average number of moves')
# plt.title('Different Game Every Time: mode ' + str(mode))
# plt.show()

	

# includeInAvg = 10
# runningAverage = runningMean(numMovesTaken, includeInAvg)

# plt.plot(numMovesTaken)
# plt.plot(runningAverage, 'r')
# plt.legend(['number of moves taken', 'sliding average ({})'.format(includeInAvg)])
# plt.xlabel('game number')
# plt.ylabel('number of moves')
# plt.title('Different Game Every Time: mode ' + str(mode))
# plt.show()

testGame = game.Game(gameX,gameY,numFood,randomSeed)

goalState = testGame.board.copy()
goalState[0,0] = -1

vis.visAllStatesGivenGoal(myRl, goalState)

vis.visNN(myNN, gameX, gameY)
