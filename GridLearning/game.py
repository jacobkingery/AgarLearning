import numpy as np
import random 

class Game(object):
	"""This is the game class. This class allows us to 
	simulate our grid-based game. 
	Here, we're defining actions as
	0: up
	1: right
	2: down
	3: left"""
	def __init__(self, x, y, numFoods, randomSeed, presetBoard=None):
		"""Here, we initialize the game state. this creates
		an x,y grid where we start the bot in the 0,0 corner
		then, we add numFoods to the grid. In this game, the
		empty cells are 0, foods are 1, and the bot is 2
		Inputs
			x: how many columns there are
			y: how many rows there are
			numFoods: how many foods there are
			randomSeed: the randomSeed to use for the game
			presetBoard: instead of generating a random board, use this board
			"""
		#So that we can gaurantee the same starting state 
		#every time
		self.randomSeed = randomSeed
		random.seed(self.randomSeed)

		self.numFoods = numFoods
		
		self.numEaten = 0
		# Because y is the number of rows and x is the number
		# of columns
		self.width = x
		self.height = y

		self.numMoves = 0

		# Get the board set up
		if presetBoard is None:
			# intialize an empty board
			self.board = -1 * np.ones((self.height,self.width))
			# set the initial condidition of the bot
			self.botPos = (0,0)
			# set the board to have the bot
			self.board[self.botPos] = 0
			# generate food randomly
			self.generateFood()
		else:
			# set the board to the inputted board
			self.board = presetBoard
			# All of our inputted boards have the bot at 0,0
			self.botPos = (0,0)

	def generateFood(self):
		"""
		Add food to the board
		"""

		#find all the empty cells in the board
		emptyRows,emptyCols = np.where(self.board == -1)

		numEmpty = len(emptyRows)
 		# if there are fewer empty spaces than the foods we have, set the number of foods to the number we can have
		if (numEmpty < self.numFoods):
			self.numFoods = numEmpty
			print ("Setting numFoods to the number of empty spaces")
		
		# Choose randomly in which of the empty cells to place the food
		foodInds = random.sample(range(numEmpty), self.numFoods)
		for foodInd in foodInds:
			# Put food in the board
			self.board[emptyRows[foodInd],emptyCols[foodInd]] = 1
		
	def printGameState(self):
		'''
		Prints the gamestate and relavant information about the game
		Inputs
		'''
		print self.board
		print "num Eaten: ", str(self.numEaten)
		print "num Left: ", str(self.numFoods - self.numEaten)

	def getPossMoves(self):
		'''
		Get the possible moves that our bot can perform
		'''
		yPos,xPos = self.botPos
		
		canMove = [
			(yPos > 0),
			(xPos < self.width - 1) ,
			(yPos < self.height - 1),
			(xPos > 0)
		]

		return [i for i,val in enumerate(canMove) if val]

	def updateGameState(self, action):
		'''
		Update the game state by performing an action
		Inputs
			action: the action, up(0), right(1), down(2), or left(3)
		Outputs
			didEat: the reward for this move
		'''
		#remember 
		#0: up, 
		#1: right, 
		#2: down, 
		#3:left
		
		#Increment the number of moves
		self.numMoves += 1

		# The rows/columns to increment/decrement given 
		# an action
		actionToChange = [
			(-1,0),
			(0,1),
			(1,0),
			(0,-1)
		]

		# Update the new bot position
		newBotPos = (
			self.botPos[0] + actionToChange[action][0], 
			self.botPos[1] + actionToChange[action][1]
		)

		#Making sure we don't go off the board
		newBotPos = (
			np.clip(newBotPos[0], 0, self.width-1),
			np.clip(newBotPos[1], 0, self.height-1)
		)

		# Check whether we ate something
		didEat = int(self.board[newBotPos] == 1)
		self.numEaten += didEat

		# if we didn't move (by trying to move into a wall) return a reward of -0.5
		if self.botPos == newBotPos:
			didEat = -0.5

		# actually update the bot position
		self.board[self.botPos] = -1
		self.board[newBotPos] = 0
		self.botPos = newBotPos

		# return whether or not we ate something (or if we ran into a wall)
		return didEat

	def isGameOver(self):
		'''
		Checks whether the game is over.
		Outputs: is the game over (boolean)
		'''
		return self.numEaten == self.numFoods

	def flattenGameState(self):
		'''
		Checks whether the game is over.
		Outputs: a flattened version of our numpy matrix for easy learning
		'''
		return self.board.flatten()[np.newaxis,:]


if __name__ == "__main__":
	# Just some tests
	game = Game(5,5,1,42)
	game.printGameState()
	moves = [1,2,2,2]
	for move in moves:
		print(game.updateGameState(move))
		game.printGameState()
		print (game.isGameOver())
		print (game.numMoves)
		