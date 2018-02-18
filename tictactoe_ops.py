###############################################################
# Author: Peizhi Yan
# Date: Feb. 18, 2017
# Description: 
#	This file is a collection of basic tic-tac-toe operations 
###############################################################
from random import randint

# Initialize Board
def initialize_board():
	return [[0,0,0],[0,0,0],[0,0,0]]

# Display Board
def display(board):
	print(' 1 2 3')
	for i in range(3):
		print(i+1, end = '')
		for j in range(3):
			if board[i][j] == 0:
				print('  ', end = '')
			if board[i][j] == -1:
				print('X ', end = '')
			if board[i][j] == 1:
				print('O ', end = '')
		print('')

# Detect Win: return: 0 for not, 1 for win, 2 for draw
def win(board):
	# Row/Column detection
	for i in range(3):
		row_sum = board[i][0] + board[i][1] + board[i][2]
		if row_sum == 3 or row_sum == -3:
			return 1
		col_sum = board[0][i] + board[1][i] + board[2][i]
		if col_sum == 3 or col_sum == -3:
			return 1
	# Diagonal detection
	left_diag_sum = board[0][0] + board[1][1] + board[2][2]
	if left_diag_sum == 3 or left_diag_sum == -3:
		return 1
	right_diag_sum = board[0][2] + board[1][1] + board[2][0]
	if right_diag_sum == 3 or right_diag_sum == -3:
		return 1
	# Detect draw
	draw = True
	for i in range(3):
		for j in range(3):
			if board[i][j] == 0:
				draw = False
				break
	if draw == True:
		return 2
	# Not win
	return 0

# Make A Move
def move(board, turn, action):
	row = int(action / 3)
	col = int(action % 3)
	if board[row][col] == 0:
		if turn == 1:
			board[row][col] = 1
		if turn == 2:
			board[row][col] = -1
	return board

# Get Flatten State
def flatten(board):
	state = [0,0,0,0,0,0,0,0,0]
	x = 0
	for i in range(3):
		for j in range(3):
			state[x] = board[i][j]
			x += 1
	return state

# A Random Move
def getRandMove(board):
	mov = randint(0,8)
	row = int(mov / 3)
	col = int(mov % 3)
	while board[row][col] != 0:
		mov = randint(0,8)
		row = int(mov / 3)
		col = int(mov % 3)
	return mov

	