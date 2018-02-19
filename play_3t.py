###############################################################
# Author: Peizhi Yan
# Date: Feb. 18, 2017
# Description: 
#	This program is to train a simple neural network model
#   play Tic-Tac-Toe by using Q-learning technique
###############################################################
import tensorflow as tf
import tictactoe_ops as game

# Game Board 3x3
board = [[0,0,0],[0,0,0],[0,0,0]] # Encoding method: 0: empty, -1: X, 1: O

# Turn: first player (1), second player (2)
turn = 1

# Hyper-parameters
BATCH_SIZE = 256
GAMMA = 0.9
EPISODES = 1000

# Placeholders
states = tf.placeholder(shape = [None, 3*3], dtype = tf.float32) # each state in a 3x3 image-like representation

# Neural Network Model
Y = tf.layers.dense(states, 500, activation = tf.nn.relu)
logits = tf.layers.dense(Y, 3*3) # the output is a 3x3 policy gradient

# Sample an action from predicted probabilities
sample = tf.argmax(logits,1)

# Game Entry
First = True
with tf.Session() as sess:
	saver = tf.train.Saver()

	# Restore the parameters
	saver.restore(sess, 'nn_saved/model.ckpt')

	available_actions = [1,1,1,1,1,1,1,1,1]

	while True:

		# AI's Turn
		state = game.flatten(board)
		action = sess.run(sample, feed_dict = {states: [state]})
		action = action[0]
		if available_actions[action] == 0:
			action = game.getRandMove(board)
		if First == True:
			First = False
			action = game.getRandMove(board)
		available_actions[action] = 0
		board = game.move(board, turn, action)
		turn = 3 - turn
		if game.win(board) == 1:
			print('AI win!')
			break
		if game.win(board) == 2:
			print('Draw!')
			break

		# Display board
		game.display(board)
		print()

		# User's Turn
		row = int(input('row:'))-1
		col = int(input('col:'))-1
		action = row * 3 + col
		board = game.move(board, turn, action)
		available_actions[action] = 0
		turn = 3 - turn
		if game.win(board) == 1:
			game.display(board)
			print('You win!')
			break
		if game.win(board) == 2:
			game.display(board)
			print('Draw!')
			break

		




