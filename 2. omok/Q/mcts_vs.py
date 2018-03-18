from utils import valid_actions, check_win
from copy import deepcopy
import numpy as np
import random
import time

import env_small as game
import env_regular
import env_large

import tree as MCTS_tree

class MCTS_vs:
    def main(self):
        env = game.GameState('pygame')
        state_size, action_size, win_mark = game.Return_BoardParams()
        agent = MCTS_tree.MCTS(win_mark)

        board_shape = [state_size, state_size]
        game_board = np.zeros(board_shape)

        do_mcts = False
        num_mcts = 1000

        # 0: O, 1: X
        turn = 0
        ai_turn = 1

        # Initialize tree root id
        root_id = (0,)

        tree = {root_id: {'board': game_board,
                          'player': turn,
                          'child': [],
                          'parent': None,
                          'n': 0,
                          'w': None,
                          'q': None}}

        while True:
            # Select action
            action = np.zeros([state_size, state_size])

            # MCTS
            if do_mcts:
                if root_id != (0,):
                    # Delete useless tree elements
                    tree_keys = list(tree.keys())
                    print('before: ' + str(len(tree_keys)))
                    for key in tree_keys:
                        if root_id != key[:len(root_id)]:
                            del tree[key]
                    print('after: ' + str(len(list(tree.keys()))))

                # Initialize Tree
                tree = {root_id: {'board': game_board,
                                  'player': turn,
                                  'child': [],
                                  'parent': None,
                                  'n': 0,
                                  'w': None,
                                  'q': None}}

                print('===================================================')
                for i in range(num_mcts):
                    # Show progress
                    if i % (num_mcts/10) == 0:
                        print('Doing MCTS: ' + str(i) + ' / ' + str(num_mcts))

                    # step 1: selection
                    leaf_id = agent.selection(tree, root_id)
                    # step 2: expansion
                    tree, child_id = agent.expansion(tree, leaf_id)
                    # step 3: simulation
                    sim_result = agent.simulation(tree, child_id)
                    # step 4: backup
                    tree = agent.backup(tree, child_id, sim_result, root_id)

                print('-------- current state --------')
                print(tree[root_id]['board'])
                q_list = {}
                actions = tree[root_id]['child']
                for i in actions:
                    q_list[root_id + (i,)] = tree[root_id + (i,)]['q']

                # Find Max Action
                max_action = max(q_list, key=q_list.get)[-1]
                max_row = int(max_action/state_size)
                max_col = int(max_action%state_size)

                print('max index: ' + '(' + str(max_row) + ' , ' + str(max_col) + ')')
                print('max action: ' + str(max_action + 1))
                do_mcts = False

                action = np.zeros([state_size * state_size])
                action[max_action] = 1
            	################################################################################################

            # Take action and get info. for update
            game_board, state, check_valid_pos, win_index, turn, coord = env.step(action)

            # If one move is done
            if check_valid_pos:
                last_action_idx = coord[0] * state_size + coord[1]

                root_id = root_id + (last_action_idx,)

            # If ai turn -> do mcts
            if turn == ai_turn:
            	do_mcts = True

            # If game is finished
            if win_index != 0:
                game_board = np.zeros([state_size, state_size])

                # Human wins!
                if turn == ai_turn:
                    ai_turn = 0
                    do_mcts = True
                else:
                    ai_turn = 1
                    do_mcts = False

            # Delay for visualization
            time.sleep(0.01)

if __name__ == '__main__':
    agent = MCTS_vs()
    agent.main()