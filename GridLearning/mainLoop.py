import game
import rl
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

myDumbNN = rl.DumbNeuralNet()

myRl = rl.ReinforcementQLearning(myDumbNN, 4, 0.5, 10, 0.99, learningRate =1, randomSeed = randomSeed)
numGames = 100
numMovesTaken = []
for i in range(numGames):
	myGame = game.Game(5,5,10,i)



	while (not myGame.isGameOver()):
		currentState = myGame.flattenGameState()
		action = myRl.getAction(currentState)
		reward = myGame.updateGameState(action)
		nextState = myGame.flattenGameState()

		myRl.storeSARS(currentState, action, reward, nextState)

		myRl.train()
	numMovesTaken.append(myGame.numMoves)

	# myGame.printGameState()

numMovesTaken = numMovesTaken

includeInAvg = 10

runningAverage = runningMean(numMovesTaken, includeInAvg)


plt.plot(numMovesTaken)
plt.plot(runningAverage, 'r')
plt.legend(['number of moves taken', 'running average'])
plt.xlabel('game number')
plt.ylabel('number of moves')
plt.show()





