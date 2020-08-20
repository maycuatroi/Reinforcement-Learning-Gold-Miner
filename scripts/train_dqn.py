import numpy as np
import gym
import sys

from rl.callbacks import ModelIntervalCheckpoint, FileLogger

sys.path.append('src')

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, BatchNormalization
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory

from Config import *
from MinerGymEnv import MinerGymEnv

env = MinerGymEnv(HOST=None, PORT=None)
env.start()
env.reset()

nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,209)))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('relu'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
# policy = GreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-2, policy=policy)
# dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.compile(Adam(), metrics=['mae'])

# dqn.load_weights('TrainedModels/dqn.h5f')

callbacks = [ModelIntervalCheckpoint('TrainedModels/dqn.h5f', interval=10000)]
# callbacks += [FileLogger(log_filename, interval=100)]

dqn.fit(env,callbacks=callbacks, nb_steps=5000000, visualize=True, verbose=1)


# dqn.save_weights('TrainedModels/dqn.h5f', overwrite=True)
# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=500, visualize=True,verbose=2)
