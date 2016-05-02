import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

import tensorflow as tf




def visStateAction (state, action):
	'''
	Visualize a state and action pair
	Plots a heatmap of the board, and an arrow for the action
	Inputs
		State
		action
	'''
	actionToChange = [
		(-1,0),
		(0,1),
		(1,0),
		(0,-1)
	]

	botRow,botCol = np.where(state == 0)

	arrowTailX = botCol[0] + 0.5
	arrowTailY = botRow[0] + 0.5

	arrowHeadX = 0.35*actionToChange[action][1]
	arrowHeadY = 0.35*actionToChange[action][0]

	
	ax = plt.axes()


	plt.pcolor(state, cmap=plt.cm.Blues)
	ax.arrow(arrowTailX, arrowTailY, arrowHeadX, arrowHeadY, head_width=0.05, head_length=0.1, fc='k', ec='k')
	ax.invert_yaxis()
	plt.show()

def visAllStatesGivenGoal (myRl, goalState):
	'''
	Visualize a state and action pair
	for any sqaure in the state that isn't food, 
	find the action we would take and plot 
	that as an arrow.

	Inputs
		myRL: the reinforcement learner
		goalState: the state we want to plot this for
	'''
	actionToChange = [
		(-1,0),
		(0,1),
		(1,0),
		(0,-1)
	]	
	# all the cells that don't ahve food
	possRow,possCol = np.where(goalState != 1)

	# create a plot
	ax = plt.axes()
	# plot the goal state
	plt.pcolor(goalState, edgecolor='k', cmap=plt.cm.Blues)
	# for each of the empty cells
	for row,col in zip(possRow,possCol):
		# make a fake game state board
		tempGameState = goalState.copy()
		tempGameState[row, col] = 0

		# get the best action
		action = myRl.getAction(tempGameState.flatten()[np.newaxis,:], evaluation=True)[0]

		# plot an arrow depicting the action
		arrowTailX = col + 0.5
		arrowTailY = row + 0.5 

		arrowHeadX = 0.35*actionToChange[action][1]
		arrowHeadY = 0.35*actionToChange[action][0]
		ax.arrow(arrowTailX, arrowTailY, arrowHeadX, arrowHeadY, head_width=0.05, head_length=0.1, fc='k', ec='k')
	
	# handle axis labels
	ax.xaxis.set_ticks(np.arange(0.5,3.5,1))
	ax.xaxis.set_ticklabels(np.arange(0,3,1))
	ax.yaxis.set_ticks(np.arange(0.5,3.5,1))
	ax.yaxis.set_ticklabels(np.arange(0,3,1))

	ax.invert_yaxis()
	plt.show()

def visNN (nn, boardX, boardY):
	'''
	Visualize weights learned by the neural network

	Inputs
		nn: our neural network
		boardX: width of board
		boardY: height of the board
	'''
	# Note: this assumes that we are playing in Mode 0 and the input is the board
	
	# get the weights
	weights =  nn.getLayerWeights(0)

	# figure out how many neurons we have
	numNeurons = weights.shape[1]

	# Set up the subplot stuff!
	# We're going to try to make a square
	subplotDim = int(round((float(numNeurons)**0.5)))
	fig = plt.figure()

	# Create a grid
	grid = AxesGrid(fig, 111, 
					nrows_ncols = (subplotDim, subplotDim),
					axes_pad=0.5,
					share_all=True,
					label_mode="L",
					cbar_location="right",
					cbar_mode="single")
	# get the min and max weights, so we can color everything the same
	maxWeight = weights.max()
	minWeight = weights.min()

	# loop over each of the neurons 
	for neuronNum,ax in enumerate(grid):
		# get learned weights
		learnedWeights = weights[:,neuronNum]
		# reshape into a square
		weightsReshaped = learnedWeights.reshape((boardY, boardX))

		# plot the weights
		mat = ax.pcolor(weightsReshaped, 
						vmin=minWeight, 
						vmax=maxWeight, 
						cmap="seismic")
		# formatting stuff
		ax.set_title('Neuron {0}'.format(neuronNum))
		ax.xaxis.set_ticks(np.arange(0.5,3.5,1))
		ax.xaxis.set_ticklabels(np.arange(0,3,1))
		ax.yaxis.set_ticks(np.arange(0.5,3.5,1))
		ax.yaxis.set_ticklabels(np.arange(0,3,1))
		ax.invert_yaxis()

	grid.cbar_axes[0].colorbar(mat)
	plt.show()