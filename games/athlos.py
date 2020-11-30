import datetime
import os

import numpy
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (5, 20, 20)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(334 + 334 + 334))  # auf 334 Felder kann gesetzt werden 
                                                          # auf 334 kann geschlagen werden
                                                          # auf 334 kann ein Anker gesetzt werden
                                                          # 2 Ankerfelder  - wirklich als eigene Aktion??
                                                          # 2 Siegfelder   - wirklich als eigene Aktion??
                                                          # Fixed list of all possible actions. You should only edit the length
                        
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # Evaluate
        self.muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "expert"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class



        ### Self-Play
        self.num_workers = 1  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 42  # - müsste mal ertestet werden aber bestimmt über 200?  ..Maximum number of moves if game is not finished before
        self.num_simulations = 200  # Number of future moves self-simulated
        self.discount = 1  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25



        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 10  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 3  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network
        
        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = []  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = []  # Define the hidden layers in the value network
        self.fc_policy_layers = []  # Define the hidden layers in the policy network



        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 100000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 10  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.005  # Initial learning rate
        self.lr_decay_rate = 1  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000



        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 42  # Number of game moves to keep for every batch element
        self.td_steps = 42  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = True  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = None  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.
        Returns:
            Positive float.
        """
        return 1


class Game(AbstractGame):
    """
    Game wrapper.
    """

    def __init__(self, seed=None):
        self.env = Connect4()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.
        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        observation, reward, done = self.env.step(action)
        return observation, reward * 10, done

    def to_play(self):
        """
        Return the current player.
        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.        
        Returns:
            An array of integers, subset of the action space.
        """
        return self.env.legal_actions()

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        return self.env.reset()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        choice = input(f"Enter the column to play for the player {self.to_play()}: ")
        while choice not in [str(action) for action in self.legal_actions()]:
            choice = input("Enter another column : ")
        return int(choice)

    def expert_agent(self):
        """
        Hard coded agent that MuZero faces to assess his progress in multiplayer games.
        It doesn't influence training
        Returns:
            Action as an integer to take in the current game state
        """
        return self.env.expert_action()

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.
        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        return f"Play column {action_number + 1}"


class Athlos:    
    def __init__(self):
        self.gap_dict = {0:6, 1:4, 2: 2, 3: 2, 4: 1, 5: 1,
                         6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0,
                         14:1, 15:1, 16:2, 17:2, 18:4, 19:6}

        #board: -9, blockiertes Feld; -2, Anker Spieler2; -1, Spieler2; 0, freies Feld; 1, Spieler1; 2 Anker Spieler1  
        self.board = numpy.zeros((20, 20), dtype="int32")        
        self.player_tiles = [[17, 3], [3, 17]]    #Spieler1, Spieler2
        self.anchor_pickup_tiles = [[8,8,1], [11,11,1]]           
        self.anchor_tiles = [[[17, 3]], [[3, 17]]]        
        self.target_tiles = [[[3,16],[4,17]], [[16,3],[17,4]]]   # [ [Sieg für Spieler1]   [Sieg für Spieler2] ] 
        self.action_points = [5, 5] #S1, S2
        
        self.tile_dict = {}  
        tile_index = 0
        for row in range(20):
            column = 0
            while column < 20 - gap_dict[row]:
                if column == 0 and gap_dict[row] != 0:
                    column += gap_dict[row]
                self.board[column][row] = 1
                self.tile_dict[tile_index] = [column, row]
                tile_index += 1
                column += 1
        self.board += 1 

        coord_dict = {item[1]: item[0] for item in tile_dict.items()}  # rückwärts-dict
        
        self.cost_array = numpy.full((20, 20), 1, dtype="int32")
        with open("tileCost.xml", "r") as cost_file:
            lines = cost_file.readlines()
            read = False
            ind = 0
            for line in enumerate(lines):
                if read == True:
                    self.cost_array[tile_dict[ind][0]][tile_dict[ind][1]] = line.replace("\t", "").replace("\n", "")
                    ind += 1  
                if "<Item>" in line:
                    read = True
                elif "</Item>" in line:
                    read = False

        self.player = 1 # Anfangsspieler wird oben in den Parametern festgelegt

        
    def to_play(self):
        return 0 if self.player == 1 else 1

    
    def reset(self):
        self.board = numpy.zeros((20, 20), dtype="int32")
        column = 0
        for row in range(20):
            while column < 20 - gap_dict[row]:
                if column == 0 and gap_dict[row] != 0:
                    column += gap_dict[row]
                self.board[row][column] = 9             # irgendetwas das Sinn macht als Wert für die Ecken?
                column += 1
        self.board -= 9                                 #   ""
        self.player = 1
        return self.get_observation()

    
#################
    def update_tiles():
        visited_tiles = {}
        components = {}
        anchor_indices = []
        
        for tile, coordinates in tile_dict.items():
            owner = board[coordinates[0], coordinates[1]]

            if len(anchor_indices < 2):
                for anch_coord in anchor_tiles[opp_index]:  #schlechte Namen für allen Ankerkram
                    if coordinates == anch_coord:
                        anchor_indices.append(tile)
            if owner != 0 and owner != self.player:
                _visited[tile] = False
                _components[tile] = -1
        
        find_components()

        opp_index = 1 if self.player == 1 else 0
        for tile, component in components.items():
            connected = False
            for anch_tile in anchor_indices:
                if components[tile] == components[anch_tile]:
                    connected = True
            if not connected:
                self.board[tile_dict[tile][0], tile_dict[tile][1]] = 0
                
                
        #####        
        def find_components():
            count = 0

            for tile, visited in visited_tiles.items():
                if not visited:
                    count += 1
                    DFS(tile, count)
        #####
        def DFS(tile, count):
            visited_tiles[tile] = True
            components[tile] = count
            coord = tile_dict[tile]
            neighbours = [
                          coord_dict[ [coord[0]+1, coord[1]  ] ],
                          coord_dict[ [coord[0]-1, coord[1]  ] ], 
                          coord_dict[ [coord[0],   coord[1]+1] ],
                          coord_dict[ [coord[0],   coord[1]-1] ]
                         ]

            for n in neighbours:
                try:
                    if visited_tiles[tile] == False:
                        DFS(n, count)

    def step(self, action):
        # hier Aktionen auf Brett anwenden
        x = tile_dict[action%334][0]
        y = tile_dict[action%334][1]
        
        action_cost = cost_array[x][y]
        player_index = 0 if self.player > 0 else 1 # 1 -> 0, -1 -> 1

        if action_points[player_index] >= action_cost:      # sollte aber eigentlich gar keine legale Aktion sein
            if action < 334:
                if self.board[x][y] == 0:
                    self.board[x][y] = self.player
                    action_points[player_index] -= action_cost

            elif action < 334*2:
                if self.board[x][y] == self.player * -1 or self.board[x][y] == (self.player + 1) * -1:
                    self.board[x][y] = 0
                    action_points[player_index] -= action_cost

            elif action  < 334*3:
                if self.board[x][y] == self.player:
                    self.board[x][y] = self.player+1 if self.player > 0 else self.player-1
                    action_points[player_index] -= action_cost
                
        done = self.have_winner() or len(self.legal_actions()) == 0

        reward = 1 if self.have_winner() else 0

        self.player *= -1

        return self.get_observation(), reward, done

    
    
    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1, 0)   # geändert von ..1.0, 0.0).. was doch gar keinen Sinn macht
        board_player2 = numpy.where(self.board == -1, 1, 0)  # oder nicht  ... vielleicht doch, siehe self_play line .. weiß ich nicht mehr
        board_to_play = numpy.full((20, 20), self.player, dtype="int32") # was ist hier wichtig? sollten auch noch ecken ausgeblendet werden? wahrscheinlich schon oder
        return numpy.array([board_player1, board_player2, board_to_play])

    
    
    def legal_actions(self):
        legal = []
        p1_tiles = []
        for y in range(20):
            for x in range(20):
                if board[x][y] != -1:
                    if board[x][y] == 1 or board[x][y] == 2:
                        p1_tiles.Add([x,y])
        p2_tiles = []
        for y in range(20):
            for x in range(20):
                if board[x][y] != :
                    if board[x][y] == -1 or board[x][y] == -2:
                        p2_tiles.Add([x,y])
        
        if self.player == 1:
            for tile in p1_tiles:
                legal.append(tile)
                legal.append(tile+668)
            for tile2 in p2_tiles:
                legal.append(tile2+334)
        else:
            for tile in p2_tiles:
                legal.append(tile)
                legal.append(tile+668)
            for tile2 in p1_tiles:
                legal.append(tile2+334)

        return legal

        
    def have_winner(self):
        for tile in self.target_tiles:
            if self.board[tile[0]][tile[1]] == self.player * -1:
                return True
        return False

    
    def expert_action(self):
        board = self.board
        player_index = 0 if player > 0 else 1
        expert_action = numpy.random.choice(self.legal_actions())
        distance = 50
        #immer Feld das am nächsten zum Gegner liegt als "Experte" ?
        for action in self.legal_actions():
            if math.sqrt((tile_dict[action%334][0] - player_tiles[player_index][0])**2
                       + (tile_dict[action%334][1] - player_tiles[player_index][1])**2)  <  distance:
                expert_action = action

        return expert_action

    
    def render(self):
        print(self.board[::-1])
