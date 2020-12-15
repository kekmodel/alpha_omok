import logging
import pickle
import random
import time
from concurrent import futures
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import model
import utils
import agents

# env_small: 9x9, env_regular: 15x15
from env import env_small as game


logging.basicConfig(
    filename='logs/log_{}.txt'.format(datetime.now().strftime('%y%m%d')),
    level=logging.INFO)

mp.set_start_method('spawn', force=True)

# Game
BOARD_SIZE = game.Return_BoardParams()[0]
N_MCTS = 200
TAU_THRES = 6
# SEED = 5
PRINT_SELFPLAY = True

# Net
N_BLOCKS = 18
IN_PLANES = 5  # history * 2 + 1
OUT_PLANES = 256

# Training
USE_TENSORBOARD = False
N_PROCESS = 2
TOTAL_ITER = 1000
MEMORY_SIZE = 4000 // N_PROCESS
BATCH_SIZE = 32
LR = 6e-4
L2 = 1e-4

# Hyperparameter sharing
agents.PRINT_MCTS = PRINT_SELFPLAY

# Set gpu or cpu
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('cuda:', use_cuda)

# Numpy printing style
np.set_printoptions(suppress=True)

# Set random seeds
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# if use_cuda:
#     torch.cuda.manual_seed_all(SEED)

# Global variables
step = 0
start_iter = 0
total_epoch = 0
result = {'Black': 0, 'White': 0, 'Draw': 0}
if USE_TENSORBOARD:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter()

# Initialize agent & model
agent = agents.ZeroAgent(BOARD_SIZE,
                         N_MCTS,
                         IN_PLANES,
                         noise=True)
agent.model = model.PVNet(N_BLOCKS,
                          IN_PLANES,
                          OUT_PLANES,
                          BOARD_SIZE).to(device)
agent.model.share_memory()

no_decay = ['bn', 'bias']
model_parameters = [
    {'params': [p for n, p in agent.model.named_parameters() if not any(
        nd in n for nd in no_decay)], 'weight_decay': L2},
    {'params': [p for n, p in agent.model.named_parameters() if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = optim.SGD(model_parameters, momentum=0.9, lr=LR)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, TOTAL_ITER)
# optimizer = optim.Adam(Agent.model.parameters(), lr=LR, eps=1e-6)

logging.info(
    '\nCUDA: {}'
    '\nAGENT: {}'
    '\nMODEL: {}'
    '\nBOARD_SIZE: {}'
    '\nN_MCTS: {}'
    '\nTAU_THRES: {}'
    '\nN_BLOCKS: {}'
    '\nIN_PLANES: {}'
    '\nOUT_PLANES: {}'
    '\nMEMORY_SIZE: {}'
    '\nBATCH_SIZE: {}'
    '\nLR: {}'
    '\nL2: {}'.format(
        use_cuda,
        type(agent).__name__,
        type(agent.model).__name__,
        BOARD_SIZE,
        N_MCTS,
        TAU_THRES,
        N_BLOCKS,
        IN_PLANES,
        OUT_PLANES,
        MEMORY_SIZE,
        BATCH_SIZE,
        LR,
        L2))


def self_play(agent, cur_memory, rank=0):
    agent.model.eval()
    state_black = deque()
    state_white = deque()
    pi_black = deque()
    pi_white = deque()
    episode = 0
    while True:
        if (episode + 1) % 10 == 0:
            logging.info('Playing Episode {:3}'.format(episode + 1))

        env = game.GameState('text')
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), 'float')
        turn = 0
        root_id = (0,)
        win_index = 0
        time_steps = 0
        action_index = None

        while win_index == 0:
            if PRINT_SELFPLAY and rank == 0:
                utils.render_str(board, BOARD_SIZE, action_index)

            # ====================== start MCTS ============================ #

            if time_steps < TAU_THRES:
                tau = 1
            else:
                tau = 0

            pi = agent.get_pi(root_id, tau, rank)

            # ===================== collect samples ======================== #

            state = utils.get_state_pt(root_id, BOARD_SIZE, IN_PLANES)

            if turn == 0:
                state_black.appendleft(state)
                pi_black.appendleft(pi)
            else:
                state_white.appendleft(state)
                pi_white.appendleft(pi)

            # ======================== get action ========================== #

            action, action_index = utils.get_action(pi)
            root_id += (action_index,)

            # ====================== print evaluation ====================== #

            if PRINT_SELFPLAY and rank == 0:
                with torch.no_grad():
                    state_input = torch.tensor([state]).to(device).float()
                    p, v = agent.model(state_input)
                    p = p.cpu().numpy()[0]
                    v = v.item()

                    print('\nPi:\n{}'.format(pi.reshape(BOARD_SIZE, BOARD_SIZE).round(decimals=2)))
                    print('\nPolicy:\n{}'.format(p.reshape(BOARD_SIZE, BOARD_SIZE).round(decimals=2)))

                if turn == 0:
                    print("\nBlack's win%: {:.2f}%".format((v + 1) / 2 * 100))
                else:
                    print("\nWhite's win%: {:.2f}%".format((v + 1) / 2 * 100))

            # =========================== step ============================= #

            board, _, win_index, turn, _ = env.step(action)
            time_steps += 1

            # ========================== result ============================ #

            if win_index != 0:
                if win_index == 1:
                    reward_black = 1.
                    reward_white = -1.
                    result['Black'] += 1

                elif win_index == 2:
                    reward_black = -1.
                    reward_white = 1.
                    result['White'] += 1

                else:
                    reward_black = 0.
                    reward_white = 0.
                    result['Draw'] += 1

            # ====================== store in memory ======================= #

                while state_black or state_white:
                    if state_black:
                        cur_memory.append((state_black.pop(),
                                           pi_black.pop(),
                                           reward_black))
                    if state_white:
                        cur_memory.append((state_white.pop(),
                                           pi_white.pop(),
                                           reward_white))

            # =========================  result  =========================== #

                if PRINT_SELFPLAY and rank == 0:
                    utils.render_str(board, BOARD_SIZE, action_index)

                    bw, ww, dr = result['Black'], result['White'], \
                        result['Draw']
                    print('')
                    print('=' * 20,
                          " {:3} Game End   ".format(episode + 1),
                          '=' * 20)
                    print('Black Win: {:3}   '
                          'White Win: {:3}   '
                          'Draw: {:2}   '
                          'Win%: {:.2f}%'.format(
                              bw, ww, dr,
                              (bw + 0.5 * dr) / (bw + ww + dr) * 100))
                    print('current memory size:', len(cur_memory))
                episode += 1
                agent.reset()
                if len(cur_memory) >= MEMORY_SIZE:
                    return utils.augment_dataset(cur_memory, BOARD_SIZE)


def train(agent, rep_memory, optimizer, scheduler):
    global step, total_epoch, writer

    agent.model.train()
    loss_all = []
    loss_v = []
    loss_p = []
    # train_memory = []
    # train_memory.extend(
    #     random.sample(rep_memory, BATCH_SIZE * len(cur_memory)))

    dataloader = DataLoader(rep_memory,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            pin_memory=use_cuda)

    print('=' * 58)
    print(' ' * 20 + ' Start Learning ' + ' ' * 20)
    print('=' * 58)
    # print('current memory size:', len(cur_memory))
    print('replay memory size:', len(rep_memory))
    # print('train memory size:', len(train_memory))
    print('optimizer: {}'.format(optimizer))
    logging.info('=' * 58)
    logging.info(' ' * 20 + ' Start Learning ' + ' ' * 20)
    logging.info('=' * 58)
    # logging.info('current memory size: {}'.format(len(cur_memory)))
    logging.info('replay memory size: {}'.format(len(rep_memory)))
    # logging.info('train memory size: {}'.format(len(train_memory)))
    logging.info('optimizer: {}'.format(optimizer))

    for s, pi, z in dataloader:
        s_batch = s.to(device).float()
        pi_batch = pi.to(device).float()
        z_batch = z.to(device).float()

        p_batch, v_batch = agent.model(s_batch)

        v_loss = (v_batch - z_batch).pow(2).mean()
        p_loss = -(pi_batch * p_batch.log()).sum(dim=-1).mean()
        loss = v_loss + p_loss

        if PRINT_SELFPLAY:
            loss_v.append(v_loss.item())
            loss_p.append(p_loss.item())
            loss_all.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1

        if USE_TENSORBOARD:
            writer.add_scalar('Loss', loss.item(), step)
            writer.add_scalar('Loss_V', v_loss.item(), step)
            writer.add_scalar('Loss_P', p_loss.item(), step)

        if PRINT_SELFPLAY:
            print('{:4} Step Loss: {:.4f}   '
                  'Loss V: {:.4f}   '
                  'Loss P: {:.4f}'.format(step,
                                          loss.item(),
                                          v_loss.item(),
                                          p_loss.item()))
    scheduler.step()
    total_epoch += 1

    if PRINT_SELFPLAY:
        print('-' * 58)
        print('{:2} Epoch Loss: {:.4f}   '
              'Loss V: {:.4f}   '
              'Loss P: {:.4f}'.format(total_epoch,
                                      np.mean(loss_all),
                                      np.mean(loss_v),
                                      np.mean(loss_p)))
    logging.info('{:2} Epoch Loss: {:.4f}   '
                 'Loss_V: {:.4f}   '
                 'Loss_P: {:.4f}'.format(total_epoch,
                                         np.mean(loss_all),
                                         np.mean(loss_v),
                                         np.mean(loss_p)))


def save_model(agent, optimizer, scheduler, datetime, n_iter, step):
    tar = {'state_dict': agent.model.state_dict(),
           'optimizer': optimizer.state_dict(),
           'scheduler': scheduler.state_dict()}
    torch.save(tar, 'data/{}_{}_{}_step_model.pickle'.format(datetime, n_iter, step))


def save_dataset(memory, datetime, n_iter, step):
    with open('data/{}_{}_{}_step_dataset.pickle'.format(datetime, n_iter, step), 'wb') as f:
        pickle.dump(memory, f, pickle.HIGHEST_PROTOCOL)


def load_model(agent, optimizer, scheduler, model_path):
    global step, start_iter
    if model_path is not None:
        print('load model: {}'.format(model_path))
        logging.info('load model: {}'.format(model_path))
        state = agent.model.state_dict()
        state.update(torch.load(model_path)['state_dict'])
        agent.model.load_state_dict(state)
        optimizer.load_state_dict(torch.load(model_path)['optimizer'])
        scheduler.load_state_dict(torch.load(model_path)['scheduler'])
        step = int(model_path.split('_')[2])
        start_iter = int(model_path.split('_')[1]) + 1


def reset_iter(result):
    result['Black'] = 0
    result['White'] = 0
    result['Draw'] = 0


def main():
    # ====================== self-play & training ====================== #
    model_path = None
    if model_path is not None:
        load_model(agent, optimizer, scheduler, model_path)

    for n_iter in range(start_iter, TOTAL_ITER):
        print('=' * 58)
        print(' ' * 20 + '  {:2} Iteration  '.format(n_iter) + ' ' * 20)
        print('=' * 58)
        logging.info(datetime.now().isoformat())
        logging.info('=' * 58)
        logging.info(' ' * 20 + "  {:2} Iteration  ".format(n_iter) + ' ' * 20)
        logging.info('=' * 58)
        datetime_now = datetime.now().strftime('%y%m%d')
        train_memory = []
        cur_memory = deque()

        with futures.ProcessPoolExecutor(max_workers=N_PROCESS) as executor:
            fs = [executor.submit(self_play, agent, cur_memory, i) for i in range(N_PROCESS)]
            for f in futures.as_completed(fs):
                train_memory.extend(f.result())

        train(agent, train_memory, optimizer, scheduler)

        save_model(agent, optimizer, scheduler, datetime_now, n_iter, step)
        save_dataset(train_memory, datetime_now, n_iter, step)

        reset_iter(result)


if __name__ == '__main__':
    main()
