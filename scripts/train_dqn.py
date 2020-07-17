import numpy as np
import gym
import sys

from rl.callbacks import ModelIntervalCheckpoint, FileLogger

sys.path.append('src')

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

from Config import *
from MinerGymEnv import MinerGymEnv

env = MinerGymEnv(HOST=None, PORT=None)
env.start()
env.reset()

nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1, 209)))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=1000000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, nb_steps_warmup=50000, gamma=.99,
               target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

# dqn.load_weights('TrainedModels/dqn.h5f')

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

callbacks = [ModelIntervalCheckpoint('TrainedModels/dqn.h5f', interval=250000)]
# callbacks += [FileLogger(log_filename, interval=100)]

dqn.fit(env,callbacks=callbacks, nb_steps=5000000, visualize=False, verbose=1)

# After training is done, we save the final weights.
# dqn.save_weights('TrainedModels/dqn.h5f', overwrite=True)
# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=500, visualize=True)
