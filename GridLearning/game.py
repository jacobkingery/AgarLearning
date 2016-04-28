import numpy as np
import random 

class Game(object):
	"""This is the game class. This class allows us to 
	simulate our grid-based game 
	Here, we're defining actions as
	0: up
	1: right
	2: down
	3: left"""
	def __init__(self, x, y, numFoods, randomSeed):
		"""Here, we initialize the game state. this creates
		an x,y grid where we start the bot in the 0,0 corner
		then, we add numFoods to the grid. In this game, the
		empty cells are 0, foods are 1, and the bot is 2"""
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
		self.board = -1 * np.ones((self.height,self.width))
		# set the initial condidition of the bot
		self.botPos = (0,0)
		self.board[0,0] = 0

		self.generateFood()

		self.numMoves = 0


	def generateFood(self):
		# this returns two numpy arrays, one corresponding to
		# rows, the other two columns
		emptyRows,emptyCols = np.where(self.board == -1)

		numEmpty = len(emptyRows)
 
		if (numEmpty < self.numFoods):
			self.numFoods = numEmpty
			print ("Setting numFoods to the number of empty spaces")
		
		foodInds = random.sample(range(numEmpty), self.numFoods)
		for foodInd in foodInds:
			self.board[emptyRows[foodInd],emptyCols[foodInd]] = 1

	def printGameState(self):
		print self.board
		print "num Eaten: ", str(self.numEaten)
		print "num Left: ", str(self.numFoods - self.numEaten)

	def getPossMoves(self):
		yPos,xPos = self.botPos
		
		canMove = [
			(yPos > 0),
			(xPos < self.width - 1) ,
			(yPos < self.height - 1),
			(xPos > 0)
		]

		return [i for i,val in enumerate(canMove) if val]

	def updateGameState(self, action):
		#remember 
		#0: up, 
		#1: right, 
		#2: down, 
		#3:left
		self.numMoves += 1
		# The rows/columns to increment/decrement given 
		# an action
		actionToChange = [
			(-1,0),
			(0,1),
			(1,0),
			(0,-1)
		]

		newBotPos = (
			self.botPos[0] + actionToChange[action][0], 
			self.botPos[1] + actionToChange[action][1]
		)

		#Making sure we don't go off the board
		newBotPos = (
			np.clip(newBotPos[0], 0, self.width-1),
			np.clip(newBotPos[1], 0, self.height-1)
		)

		didEat = int(self.board[newBotPos] == 1)
		self.numEaten += didEat

		if self.botPos == newBotPos:
			didEat = -1

		self.board[self.botPos] = -1
		self.board[newBotPos] = 0
		self.botPos = newBotPos

		return didEat

	def isGameOver(self):
		return self.numEaten == self.numFoods

	def flattenGameState(self):
		return self.board.flatten()[np.newaxis,:]


if __name__ == "__main__":
	game = Game(5,5,1,42)
	game.printGameState()
	moves = [1,2,2,2]
	for move in moves:
		print(game.updateGameState(move))
		game.printGameState()
		print (game.isGameOver())
		print (game.numMoves)
		