import sys
import numpy as np
from GAME_SOCKET_DUMMY import GameSocket  # in testing version, please use GameSocket instead of GAME_SOCKET_DUMMY
from MINER_STATE import State

TreeID = 1
TrapID = 2
SwampID = 3


class MinerEnv:
    def __init__(self, host, port):
        self.socket = GameSocket(host, port)
        self.state = State()
        self.score_pre = self.state.score  # Storing the last score for designing the reward function

    def start(self):  # connect to server
        self.socket.connect()

    def end(self):  # disconnect server
        self.socket.close()

    def send_map_info(self, request):  # tell server which map to run
        self.socket.send(request)

    def reset(self):  # start new game
        try:
            message = self.socket.receive()  # receive game info from server
            self.state.init_state(message)  # init state
        except Exception as e:
            import traceback
            traceback.print_exc()

    def step(self, action):  # step process
        self.socket.send(action)  # send action to server
        try:
            message = self.socket.receive()  # receive new state from server
            self.state.update_state(message)  # update to local state
        except Exception as e:
            import traceback
            traceback.print_exc()

    # Functions are customized by client
    def get_state(self):
        # Building the map
        # view = np.zeros([self.state.mapInfo.max_x + 1, self.state.mapInfo.max_y + 1], dtype=int)
        view = np.zeros(shape=[self.state.mapInfo.max_x+2,self.state.mapInfo.max_y+2,5],dtype=float)
        status_view= []
        for i in range(self.state.mapInfo.max_x+1):
            for j in range(self.state.mapInfo.max_y+1):
                if self.state.mapInfo.get_obstacle(i, j) == TreeID:  # Tree
                    view[i, j,0] = -TreeID
                if self.state.mapInfo.get_obstacle(i, j) == TrapID:  # Trap
                    view[i, j,0] = -TrapID
                if self.state.mapInfo.get_obstacle(i, j) == SwampID:  # Swamp
                    view[i, j,0] = -SwampID
                if self.state.mapInfo.gold_amount(i, j) > 0:
                    view[i, j,1] = self.state.mapInfo.gold_amount(i, j)
        view[self.state.x, self.state.y,3] = 1

        # Add position and energy of agent to the DQNState

        next_round_energy,player_free_count = self.get_next_round_engergy()
        # status_view.append([self.state.score,self.state.energy,next_round_energy]) #Current user state

        status_view.append([player_free_count/10,self.state.energy/50.,next_round_energy/100]) #Current user state

        # Add position of bots
        for player in self.state.players:
            if player["playerId"] != self.state.id:
                view[player["posx"], player["posy"],2] = 1
                # DQNState.append(player["posx"])
                # DQNState.append(player["posy"])
                energy = 0
                # score = 0
                free_count = 0
                if 'energy' in player:
                    energy = player["energy"]
                if 'score' in player:
                    score = player["score"]
                if 'free_count' in player:
                    free_count = player["free_count"]
                status_view.append([free_count,energy,0])

        # Convert the DQNState from list to array for training
        # DQNState = np.array(DQNState)
        status_view = np.array(status_view)
        h,w = status_view.shape
        status_view[:,0]/= 10.
        status_view[:,1]/= 50.
        view[:h,:w,4] = status_view
        view[:, :,0] /= 3
        view[:, :,1] /= view[:,:,1].max()
        return view

    def get_next_round_engergy(self):
        free_count = 0
        for p in self.state.players:
            if p['playerId'] == self.state.id:
                free_count = p['freeCount']
        next_e = self.state.energy
        for i in range(4 - free_count):
            next_e += next_e / max(i, 1)
        # return next_e,free_count
        return 0,free_count

    def dig_score(self):
        pass

    def get_reward(self):
        # Calculate reward
        reward = 0
        score_action = self.state.score  # - self.score_pre
        self.score_pre = self.state.score
        if score_action > 0:
            # If the DQN agent crafts golds, then it should obtain a positive reward (equal score_action)
            reward += score_action
            # print('Craft gold : {}'.format(score_action))
        next_e ,player_free_count= self.get_next_round_engergy()
        if next_e <= 0:  # Do not stand while you have full energy :(
            reward -= 10

        if next_e >= 1 and self.state.lastAction == 4:
            reward-=10

        # If the DQN agent crashs into obstacels (Tree, Trap, Swamp), then it should be punished by a negative reward
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TreeID:  # Tree
            reward -= TreeID
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == TrapID:  # Trap
            reward -= TrapID
        if self.state.mapInfo.get_obstacle(self.state.x, self.state.y) == SwampID:  # Swamp
            reward -= SwampID

        # If out of the map, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_WENT_OUT_MAP:
            reward += -10
        # Run out of energy, then the DQN agent should be punished by a larger nagative reward.
        if self.state.status == State.STATUS_ELIMINATED_OUT_OF_ENERGY:
            reward += -10
        return reward

    def check_terminate(self):
        # Checking the status of the game
        # it indicates the game ends or is playing
        return self.state.status != State.STATUS_PLAYING
