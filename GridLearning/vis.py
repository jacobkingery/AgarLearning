import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np

import tensorflow as tf




def visStateAction (state, action):
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
	actionToChange = [
		(-1,0),
		(0,1),
		(1,0),
		(0,-1)
	]	

	possRow,possCol = np.where(goalState != 1)

	ax = plt.axes()
	plt.pcolor(goalState, edgecolor='k', cmap=plt.cm.Blues)

	for row,col in zip(possRow,possCol):
		tempGameState = goalState.copy()
		tempGameState[row, col] = 0

		action = myRl.getAction(tempGameState.flatten()[np.newaxis,:], evaluation=True)[0]

		arrowTailX = col + 0.5
		arrowTailY = row + 0.5 

		arrowHeadX = 0.35*actionToChange[action][1]
		arrowHeadY = 0.35*actionToChange[action][0]
		ax.arrow(arrowTailX, arrowTailY, arrowHeadX, arrowHeadY, head_width=0.05, head_length=0.1, fc='k', ec='k')
	
	ax.xaxis.set_ticks(np.arange(0.5,3.5,1))
	ax.xaxis.set_ticklabels(np.arange(0,3,1))
	ax.yaxis.set_ticks(np.arange(0.5,3.5,1))
	ax.yaxis.set_ticklabels(np.arange(0,3,1))

	ax.invert_yaxis()
	plt.show()

def visNN (nn, boardX, boardY):
	# Note: this assumes that we are playing in Mode 0 and the input is the board
	
	weights =  nn.getLayerWeights(0)

	numNeurons = weights.shape[1]

	# Set up the subplot stuff!
	# We're going to try to make a square
	subplotDim = int(round((float(numNeurons)**0.5)))
	fig = plt.figure()

	# f, axarr = plt.subplots(subplotDim, subplotDim, figsize=(10, 10))
	grid = AxesGrid(fig, 111, 
					nrows_ncols = (subplotDim, subplotDim),
					axes_pad=0.5,
					share_all=True,
					label_mode="L",
					cbar_location="right",
					cbar_mode="single")
	maxWeight = weights.max()
	minWeight = weights.min()

	for neuronNum,ax in enumerate(grid):
		learnedWeights = weights[:,neuronNum]
		weightsReshaped = learnedWeights.reshape((boardY, boardX))

		mat = ax.pcolor(weightsReshaped, 
						vmin=minWeight, 
						vmax=maxWeight, 
						cmap="seismic")
		ax.set_title('Neuron {0}'.format(neuronNum))
		ax.xaxis.set_ticks(np.arange(0.5,3.5,1))
		ax.xaxis.set_ticklabels(np.arange(0,3,1))
		ax.yaxis.set_ticks(np.arange(0.5,3.5,1))
		ax.yaxis.set_ticklabels(np.arange(0,3,1))
		ax.invert_yaxis()
	grid.cbar_axes[0].colorbar(mat)

	plt.show()


	# for neuronNum in range(numNeurons):
	# 	subplotRow = neuronNum/subplotDim
	# 	subplotCol = neuronNum % subplotDim
	# 	ax = axarr[subplotRow, subplotCol]
	# 	learnedWeights = weights[:,neuronNum]
	# 	weightsReshaped = learnedWeights.reshape((boardY, boardX))

	# 	mat = ax.pcolor(weightsReshaped, vmin=minWeight, vmax=maxWeight)
	# 	ax.set_title('Neuron {0}'.format(neuronNum))
	# 	# plt.colorbar(mat)
	# 	# ax.colorbar()
	# f.tight_layout()
	# plt.show()

	# cax = f.add_axes([0.9, 0.1, 0.03, 0.8])
	# f.colorbar(mat, cax=cax)

	#Now we need to slice this appropriately for each of the neurons. This returns

	# plt.matshow(x)
	# plt.colorbar()


