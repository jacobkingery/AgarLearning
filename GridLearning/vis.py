import matplotlib.pyplot as plt
import numpy as np




def visStateAction (state, action):
	actionToChange = [
		(-1,0),
		(0,1),
		(1,0),
		(0,-1)
	]

	botRow,botCol = np.where(state == 2)

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
		tempGameState[col, row] = 2
		print tempGameState
		action = myRl.getAction(tempGameState.flatten()[np.newaxis,:], evaluation=True)
		
		arrowTailX = col + 0.5
		arrowTailY = row + 0.5

		arrowHeadX = 0.35*actionToChange[action][1]
		arrowHeadY = 0.35*actionToChange[action][0]
		ax.arrow(arrowTailX, arrowTailY, arrowHeadX, arrowHeadY, head_width=0.05, head_length=0.1, fc='k', ec='k')

	ax.invert_yaxis()
	plt.show()



