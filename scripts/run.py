import sys
sys.path.append('src')
from MinerGymEnv import MinerGymEnv
from Model.DQNModel import DQN # A class of creating a deep q-learning model
from MinerEnv import MinerEnv # A class of creating a communication environment between the DQN model and the GameMiner environment (GAME_SOCKET_DUMMY.py)
from Memory import Memory # A class of creating a batch in order to store experiences for the training process

import pandas as pd
import datetime
import numpy as np


HOST = "localhost"
PORT = 1111
if len(sys.argv) == 3:
    HOST = str(sys.argv[1])
    PORT = int(sys.argv[2])

# Create header for saving DQN learning file
now = datetime.datetime.now() #Getting the latest datetime
header = ["Ep", "Step", "Reward", "Total_reward", "Actdion", "Epsilon", "Done", "Termination_Code"] #Defining header for the save file
filename = "Data/data_" + now.strftime("%Y%m%d-%H%M") + ".csv"
with open(filename, 'w') as f:
    pd.DataFrame(columns=header).to_csv(f, encoding='utf-8', index=False, header=True)

# Parameters for training a DQN model
N_EPISODE = 10000 #The number of episodes for training
MAX_STEP = 1000   #The number of steps for each episode
BATCH_SIZE = 32   #The number of experiences for each replay
MEMORY_SIZE = 100000 #The size of the batch for storing experiences
SAVE_NETWORK = 300  # After this number of episodes, the DQN model is saved for testing later.
INITIAL_REPLAY_SIZE = 10000000 #The number of experiences are stored in the memory batch before starting replaying
INPUTNUM = 1100 #The number of input values for the DQN model
ACTIONNUM = 6  #The number of actions output from the DQN model
MAP_MAX_X = 21 #Width of the Map
MAP_MAX_Y = 9  #Height of the Map
DEBUG = 1

# Initialize a DQN model and a memory batch for storing experiences
DQNAgent = DQN(INPUTNUM, ACTIONNUM)
DQNAgent.load_model('TrainedModels/','DQNmodel_20200820-0538_ep7500')
# DQNAgent.target_model.load_weights('TrainedModels/DQNmodel_20200819-0726_ep5400.h5')
memory = Memory(MEMORY_SIZE)

# Initialize environment
minerEnv = MinerGymEnv(HOST, PORT,debug=DEBUG) #Creating a communication environment between the DQN model and the game environment (GAME_SOCKET_DUMMY.py)
minerEnv.start()  # Connect to the game

train = False #The variable is used to indicate that the replay starts, and the epsilon starts decrease.
#Training Process
#the main part of the deep-q learning agorithm
for episode_i in range(0, N_EPISODE):
    try:
        # Choosing a map in the list
        mapID = np.random.randint(1, 6) #Choosing a map ID from 5 maps in Maps folder randomly
        posID_x = np.random.randint(MAP_MAX_X) #Choosing a initial position of the DQN agent on X-axes randomly
        posID_y = np.random.randint(MAP_MAX_Y) #Choosing a initial position of the DQN agent on Y-axes randomly
        #Creating a request for initializing a map, initial position, the initial energy, and the maximum number of steps of the DQN agent
        request = ("map" + str(mapID) + "," + str(posID_x) + "," + str(posID_y) + ",50,100")
        #Send the request to the game environment (GAME_SOCKET_DUMMY.py)
        minerEnv.send_map_info(request)

        # Getting the initial state
        minerEnv.reset() #Initialize the game environment
        s = minerEnv.get_state()#Get the state after reseting.
        #This function (get_state()) is an example of creating a state for the DQN model
        total_reward = 0 #The amount of rewards for the entire episode
        terminate = False #The variable indicates that the episode ends
        maxStep = minerEnv.state.mapInfo.maxStep #Get the maximum number of steps for each episode in training
        #Start an episde for training
        for step in range(0, maxStep):
            action = DQNAgent.act(s)  # Getting an action from the DQN model from the state (s)
            minerEnv.step(str(action))  # Performing the action in order to obtain the new state
            s_next = minerEnv.get_state()  # Getting a new state
            reward = minerEnv.get_reward()  # Getting a reward
            terminate = minerEnv.check_terminate()  # Checking the
            if terminate == True:
                #If the episode ends, then go to the next episode
                break
    except:
        print('z')