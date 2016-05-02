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
	'''
	Calculates the running mean for an array
	Inputs:
		arr: the array to calculate the running mean over
		numPoints: how many points to include in the running mean
	Outputs:
		An array of the running mean
	'''
	runningAvg = []
	for i,val in enumerate(arr):
		
		if i < numPoints:
			avg = float(sum(arr[0:i+1]))/(i+1)
		else:
			avg = float(sum(arr[i-numPoints+1:i+1]))/(numPoints)

		runningAvg.append(avg)

	return runningAvg

# Initial parameters 
randomSeed = 42
gameX = 3
gameY = 3
numFood = 1
numActions = 4
bpLength = 10

def findPossibleBoards(gameX, gameY, numFood):
	'''
	Finds all possible game boards (for evaluation)
	Inputs: 
		gameX: the width of the game board
		gameY: the height of the game board
		numFood: the number of food to include in the game
	Outputs:
		A list of possible game boards (numpy arrays)
	'''
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
# get all the possible eval boards
evalBoards = findPossibleBoards(gameX, gameY, numFood)

def constDiscountFactor(i):
	'''
	A constant discount factor
	'''
	return 0.5

def incDiscountFactor(i):
	'''
	An increasing discount factor
	'''
	initial = 0.2
	final = 0.5
	rate = 0.001
	return min(initial + i*rate, final)

def constExpRate(i):
	'''
	A constant exploratory rate
	'''
	return 0.2

def decayExpRate(i):
	'''
	A decaying exploratory rate
	'''
	N = 0.9
	tau = 5000.0
	return N * np.exp(-i / tau)

def constLearningRate(i):
	'''
	A constant learning rate
	'''
	return .9
def decayLearningRate(i):
	'''
	A decaying learning rate
	'''
	N = 0.9
	tau = 2000.0
	return N * np.exp(-i / tau)

# Parameters we need to pass into our neural net
stateSize = gameX * gameY
hiddenLayers = [(9, tf.tanh)]
mode = 0

# Initalize neural net and reinforcement learner
myNN = nn.NeuralNet(stateSize, numActions, hiddenLayers, mode=mode)
myRl = rl.ReinforcementQLearning(myNN, numActions, constExpRate, bpLength, constDiscountFactor, constLearningRate, randomSeed=randomSeed)

# Variables to keep track of our performance
numGames = 10000
numMovesTaken = []
numGamesEvalIterationList = []
numGamesEvalAverageList = []
numGamesEvalMedianList = []
print('Playing {0} games'.format(str(numGames)))
# Play a bunch of games (tqdm makes pretty progress bars)
for i in tqdm.tqdm(range(numGames)):
	# initialize game
	myGame = game.Game(gameX,gameY,numFood,i)
	# Play the game
	while (not myGame.isGameOver()):
		# Get the current state
		currentState = myGame.flattenGameState()
		# Get the best new action
		actionExpPair = myRl.getAction(currentState, i)
		action = actionExpPair[0]
		exp = actionExpPair[1]
		# Get the reward of performing that actions
		reward = myGame.updateGameState(action) #+ 0.1 * int(moveCloser)
		# Get the next state
		nextState = myGame.flattenGameState()

		# Store the state, action, reward, state tuple
		myRl.storeSARS(currentState, action, reward, nextState)
		# Train the learner
		myRl.train(i)

	# After a game, append the number of moves it took to a list that keeps track of this
	numMovesTaken.append(myGame.numMoves)

	# Every 50 games, perform an evaluation
	if (i+1)%50 == 0:
		numGamesEvalList = []
		# Loop through each different possible board
		for evalBoard in evalBoards:
			# Play games
			myGameEval = game.Game(gameX,gameY,numFood,1, presetBoard=evalBoard.copy())
			# Here, we're stopping the game if the number of moves climbs above 100
			while (not myGameEval.isGameOver()) and myGameEval.numMoves < 100:
				currentState = myGameEval.flattenGameState()
				action = myRl.getAction(currentState, evaluation=True)[0]
				reward = myGameEval.updateGameState(action) #+ 0.1 * int(moveCloser)
			# Append the number of moves it took
			numGamesEvalList.append(myGameEval.numMoves)
		# Append data about how far we are in the larger game
		numGamesEvalIterationList.append(i)
		# Append the mean and median of the number of moves it took to play the game
		numGamesEvalAverageList.append(np.mean(numGamesEvalList))
		numGamesEvalMedianList.append(np.median(numGamesEvalList))


# Plot a bar chart of the evaluation period
plt.bar(np.array(numGamesEvalIterationList), numGamesEvalAverageList,color='b',edgecolor = "none",width=50)
plt.xlabel('Games trained on')
plt.ylabel('Mean number of moves to complete evaluation boards')
plt.show()

# Plot the running average of number of moves it took in the training period
includeInAvg = 100
runningAverage = runningMean(numMovesTaken, includeInAvg)

plt.plot(numMovesTaken)
plt.plot(runningAverage, 'r')
plt.legend(['Number of moves taken', 'Sliding average ({})'.format(includeInAvg)])
plt.xlabel('Game number')
plt.ylabel('Number of moves')
plt.show()


# Plot a visualiztion for each of the eval boards that plots arrows showing where we would go for an cell
for evalBoard in evalBoards:
	goalState = evalBoard.copy()
	# override bot position
	goalState[0,0] = -1
	vis.visAllStatesGivenGoal(myRl, goalState)

# Visualize the weights learned by the neural network
vis.visNN(myNN, gameX, gameY)
