import game
import rl
import nn
import random
import vis

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from itertools import permutations
from itertools import combinations


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
bpLength = 10

def findPossibleBoards(gameX, gameY, numFood):
	spotList = range(gameX*gameY)[1:]
	foodPositionsList = list(combinations(spotList, numFood))
	boardList = []
	for foodPositions in foodPositionsList:
		boardFlat = np.concatenate((np.array([0]),(-1 * np.ones((gameX * gameY - 1)))))
		for foodPosition in foodPositions:
			boardFlat[foodPosition] = 1
		board = np.reshape(boardFlat, (gameX,gameY))
		boardList.append(board)
	return boardList

evalBoards = findPossibleBoards(gameX, gameY, numFood)

def constDiscountFactor(i):
	return 0.5
def incDiscountFactor(i):
	initial = 0.2
	final = 0.5
	rate = 0.001
	return min(initial + i*rate, final)

def constExpRate(i):
	return 0.2
def decayExpRate(i):
	N = 0.9
	tau = 5000.0
	return N * np.exp(-i / tau)

def constLearningRate(i):
	return .9
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
		for evalBoard in evalBoards:
			myGameEval = game.Game(gameX,gameY,numFood,1, presetBoard=evalBoard.copy())
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
