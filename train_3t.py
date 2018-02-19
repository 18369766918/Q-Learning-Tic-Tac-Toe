###############################################################
# Author: Peizhi Yan
# Date: Feb. 18, 2017
# Description: 
#	This program is to train a simple neural network model
#   play Tic-Tac-Toe by using Q-learning technique
###############################################################
import tensorflow as tf
import tictactoe_ops as game
import random

# Game Board 3x3
board = [[0,0,0],[0,0,0],[0,0,0]] # Encoding method: 0: empty, -1: X, 1: O

# Turn: first player (1), second player (2)
turn = 1

# Hyper-parameters
BATCH_SIZE = 256
GAMMA = 0.9
EPISODES = 9999
RANDOM_THRESHOLD = 0.2
THRESHOLD_DISCOUNT = 0.90

# Placeholders
states = tf.placeholder(shape = [None, 3*3], dtype = tf.float32) # each state in a 3x3 image-like representation
actions = tf.placeholder(shape = [None], dtype = tf.int32) # we encoded the actions: 0,1,2,3,4,5,6,7,8 represent 9 different positions
rewards = tf.placeholder(shape = [None], dtype = tf.float32) # discounted rewards between -1 and +1

# Neural Network Model
Y = tf.layers.dense(states, 500, activation = tf.nn.relu)
logits = tf.layers.dense(Y, 3*3) # the output is a 3x3 policy gradient

# Sample an action from predicted probabilities
sample = tf.argmax(logits,1)

# Loss function: softmax cross entropy
loss = tf.reduce_sum((-rewards) * tf.losses.softmax_cross_entropy(onehot_labels = tf.one_hot(actions, 3*3), logits = logits))

# Optimizer: RMS 
optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.001, decay = 0.99).minimize(loss)

log1 = open('loss.log','a')
log2 = open('rand_rate.log','a')

# Training Entry
with tf.Session() as sess:
	saver = tf.train.Saver()
	# first time training, need to initialize the parameters of the neural network
	sess.run(tf.global_variables_initializer())
	# Training Loop
	for episode in range(EPISODES):
		# Training Set
		S = []
		A = []
		R = []
		# Generate A Batch of Training Set Via Self-play
		while len(S) < BATCH_SIZE:
			# Reset everything
			board = game.initialize_board()
			turn = 1
			_states = []
			_actions = []
			_rewards = []
			available_actions = [1,1,1,1,1,1,1,1,1]
			# Play A Game Until End
			while True:
				#print('New Game Simulation Start!')
				# Get state
				state = game.flatten(board)
				# Get Action
				rand_trigger = random.random()
				if rand_trigger < RANDOM_THRESHOLD and episode > 1000:
					action = game.getRandMove(board)
				else:
					action = sess.run(sample, feed_dict = {states: [state]})
					action = action[0]
					if available_actions[action] == 0:
						action = game.getRandMove(board)
					available_actions[action] = 0
				# Make Move
				board = game.move(board, turn, action)
				if game.win(board) != 0:
					_states.append(state)
					_actions.append(action)
					_rewards.append(0)
					break # game over
				turn = 3 - turn
				# Collect Results
				_states.append(state)
				_actions.append(action)
				_rewards.append(0) # default reward is 0!
			# Process Rewards After Each Game
			if game.win(board) == 1:
				# Discounted Reward: initialized to 1
				r = 1
				# Apply the discount function to each (s,a) pair
				i = len(_rewards) - 1
				flag = True # to indicate winner (True) or loser (False)
				while i > 0:
					if flag == True:
						_rewards[i] = r
					else:
						_rewards[i] = -r
					r = GAMMA*r # discount
					flag = False
					i -= 1
			for i in range(len(_states)):
				S.append(_states[i])
				A.append(_actions[i])
				R.append(_rewards[i])
		# Train Neural Network
		training_batch = {states: S, actions: A, rewards: R}
		_, training_loss = sess.run([optimizer, loss], feed_dict = training_batch)
		print('Episode:',episode,'  loss:', training_loss, ' random_rate:',RANDOM_THRESHOLD)
		log1.write(str(training_loss)+'\n')
		log2.write(str(RANDOM_THRESHOLD)+'\n')
		# save the trained neural network
		if episode % 100 == 0:
			saver.save(sess, 'nn_saved/model.ckpt')
			if episode > 1000:
				RANDOM_THRESHOLD *= THRESHOLD_DISCOUNT

log1.close()
log2.close()



