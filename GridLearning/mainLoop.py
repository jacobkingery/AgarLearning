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
numFood = 3
numActions = 4
bpLength = 10

def constDiscountFactor(i):
	return 0.15
def incDiscountFactor(i):
	initial = 0.2
	final = 0.5
	rate = 0.001
	return min(initial + i*rate, final)

def constExpRate(i):
	return 0.2
def decayExpRate(i):
	N = 0.8
	tau = 5000.0
	return N * np.exp(-i / tau)

def constLearningRate(i):
	return .8
def decayLearningRate(i):
	N = 0.9
	tau = 2000.0
	return N * np.exp(-i / tau)

stateSize = gameX * gameY
hiddenLayers = [(9, tf.tanh)]
mode = 0

myNN = nn.NeuralNet(stateSize, numActions, hiddenLayers, mode=mode)
myRl = rl.ReinforcementQLearning(myNN, numActions, decayExpRate, bpLength, constDiscountFactor, constLearningRate, randomSeed=randomSeed)

numGames = 10000
numMovesTaken = []
numGamesEvalIterationList = []
numGamesEvalAverageList = []
numGamesEvalMedianList = []
print('Playing {0} games'.format(str(numGames)))
for i in tqdm.tqdm(range(numGames)):
	myGame = game.Game(gameX,gameY,numFood,i)
	botPos = myGame.botPos
	while (not myGame.isGameOver()):
		currentState = myGame.flattenGameState()
		actionExpPair = myRl.getAction(currentState, i)
		action = actionExpPair[0]
		exp = actionExpPair[1]
		moveCloser = myGame.didGetCloserToFood
		reward = myGame.updateGameState(action) + 0.1 * int(moveCloser)
		nextState = myGame.flattenGameState()

		# if not exp:
		myRl.storeSARS(currentState, action, reward, nextState)
		myRl.train(i)

		# if i == numGames-1:
		# 	myGame.printGameState()

	numMovesTaken.append(myGame.numMoves)

	if i%50 == 0 and i!=0:
		numGamesEvalList = []
		for j in range(20):
			myGameEval = game.Game(gameX,gameY,numFood,j*.11)

			while (not myGameEval.isGameOver()) and myGameEval.numMoves < 100:
				currentState = myGameEval.flattenGameState()
				action = myRl.getAction(currentState, evaluation=True)[0]

				moveCloser = myGameEval.didGetCloserToFood
				reward = myGameEval.updateGameState(action) + 0.1 * int(moveCloser)
				nextState = myGameEval.flattenGameState()
			# print myGameEval.numMoves
			numGamesEvalList.append(myGameEval.numMoves)
		numGamesEvalIterationList.append(i)
		numGamesEvalAverageList.append(np.mean(numGamesEvalList))
		numGamesEvalMedianList.append(np.median(numGamesEvalList))



plt.bar(np.array(numGamesEvalIterationList)-5, numGamesEvalAverageList,color='b',width=10)
plt.bar(np.array(numGamesEvalIterationList)+5, numGamesEvalMedianList,color='g',width=10)
plt.legend(['mean','median'])
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

testGame = game.Game(gameX,gameY,numFood,randomSeed)

goalState = testGame.board.copy()
goalState[0,0] = -1

vis.visAllStatesGivenGoal(myRl, goalState)

vis.visNN(myNN, gameX, gameY)
