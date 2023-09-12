# Edge caching & softhit system
# Xinyu Zhang in central south university
# 05/01 2023

import numpy as np
from env import env
import argparse
import datetime
from sac import SAC
from DDQN import DDQN
from replay_memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
import copy
import matplotlib.pyplot as plt

from utils import combine_agents, distribute_agents, combine_agents_reward_based, read_csv

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--alpha', type=float, default=0.1, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=3407, metavar='N',
                    help='random seed (default: 1234)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='G',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=100000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', type=bool, default=True,
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# set random seed
np.random.seed(args.seed)

edge_n = 4
edge_size = 10
end_edge = 10
end_n = end_edge * edge_n
content_n = 50
request_t = 50
c_c = 1
u_c = 10
cl_c = 5

def zipf(content_n, end_n, table, t_request, a=1):
    if len(table.shape) == 1:
        table = np.array([table])
    p = np.array([1 / (i ** a) for i in range(1, content_n + 1)])
    p = p / sum(p)
    request = np.zeros((end_n, t_request))
    for i in range(end_n):
        for j in range(t_request):
            c = np.random.choice(list(range(content_n)), 1, False, p)
            request[i, j] = table[i][c]

    return request


def main():
    """
    This version solves the discrete action
    :return:
    """
    # random table between end device and content
    table = np.array([np.random.permutation(list(range(content_n))) for _ in range(end_n)])
    # build request list for all users
    requests_list = []
    '''
    for edge in range(edge_n):
        request = zipf(content_n, 10, table, request_t + 1, a=(edge+1)/edge_n).astype(int)
        requests_list.append(request)
    '''
    requests = zipf(content_n, end_n, table, request_t + 1).astype(int)
    for i in range(edge_n):
        requests_list.append(requests[i * end_edge:(i + 1) * end_edge])

    # Build caching environment
    caching_env = env(edge_n, end_edge, edge_size, content_n)

    # GENERATE MULTIPLE CLASSES FOR RL
    agent_list = list()
    for iot in range(edge_n):
        agent_list.append(SAC(caching_env.state_n, caching_env.action_n, args))
        #agent_list.append(DDQN(caching_env.state_n, caching_env.action_n, args))
    cost_temp = 10 ** 20

    # central aggregator
    cen_agent = SAC(caching_env.state_n, caching_env.action_n, args)
    #cen_agent = DDQN(caching_env.state_n, caching_env.action_n, args)
    # Memory
    memory_list = list()
    for iot in range(edge_n):
        memory_list.append(ReplayMemory(args.replay_size, args.seed))

    # ========****** start training ******========
    RL_step = [0] * edge_n
    rewards_list = []
    episode_list = []
    for episode in range(1, args.num_steps + 1):
        # requests = zipf(content_n, end_n, table, request_t + 1).astype(int)
        # requests_list = []
        # for i in range(edge_n):
        #     requests_list.append(requests[i * end_edge:(i + 1) * end_edge])
        caching_env.reset()

        done = False
        for step in range(request_t):
            if step == request_t - 1:
                done = True
            cur_state_list = []
            action_list = []
            cur_requests = []
            provide_state_list = []
            for agent in range(edge_n):
                provide_state = caching_env.request_provide(agent, requests_list[agent][:, step])
                cur_state = caching_env.transform_state(caching_env.cache_state[agent], requests_list[agent][:, step], provide_state)
                #cur_state = caching_env.transform_state(caching_env.cache_state[agent], requests_list[agent][:, step])

                cur_state_list.append(cur_state)
                action = agent_list[agent].select_action(cur_state)
                action_list.append(action)
                provide_state_list.append(provide_state)
                cur_requests.append(requests_list[agent][:, step])

            state, rewards = caching_env.step(cur_requests, action_list, provide_state_list)
            # store experience
            next_state_list = []

            for agent in range(edge_n):
                next_state = caching_env.transform_state(caching_env.cache_state[agent],
                                                         requests_list[agent][:, step + 1], caching_env.provide_state[agent])
                next_state_list.append(next_state)
                memory_list[agent].push(cur_state_list[agent],
                                        action_list[agent],
                                        caching_env.rewards[agent],
                                        #-caching_env.cur_cost[agent],
                                        next_state,
                                        done)
            policy_loss = 0
            for agent in range(edge_n):
                if len(memory_list[agent]) > args.batch_size:
                    critic_loss, policy_loss = \
                        agent_list[agent].update_parameters(memory_list[agent],
                                                            args.batch_size,
                                                            RL_step[agent])
                    RL_step[agent] += 1

        # federated learning
        if episode == 1:
            cen_agent = combine_agents(cen_agent, agent_list)
            agent_list = distribute_agents(cen_agent, agent_list)

        if episode % 50 == 0:
            # cen_agent = combine_agents(cen_agent, agent_list)
            #cen_agent = combine_agents_reward_based(cen_agent, agent_list, caching_env.cost)
            cen_agent = combine_agents_reward_based(cen_agent, agent_list, caching_env.rewards)
            agent_list = distribute_agents(cen_agent, agent_list)

        #cost = caching_env.cost
        #rewards = caching_env.rewards
        #rewards = caching_env.get_reward()
        #state = caching_env.cache_state

        print('Current training episode: ', episode, 'Current system state', state, ' Current system reward: ',
              #sum(cost) / edge_n / request_t / end_edge)
              sum(rewards) / edge_n / request_t / end_edge, 'reward:', rewards)

        rewards_list.append(sum(rewards) / edge_n / request_t / end_edge)
        episode_list.append(episode)
        # if episode % 50 == 0:
        #     plt.plot(episode_list, rewards_list)
        #
        #     plt.show()
        #print('Current training episode:', episode, ', Current reward: ', rewards[0], rewards[1], rewards[2], rewards[3])
        # Save model
        if episode > 2000 and episode % 100 == 0 and sum(caching_env.rewards) < cost_temp:
            cost_temp = sum(caching_env.rewards)
            for agent in range(edge_n):
                agent_list[agent].save_model(agent)
    plt.plot(episode_list, rewards_list)
    data = np.column_stack((episode_list, rewards_list))
    np.savetxt('result/data.csv', data, delimiter=',', header="Episodes,Rewards", comments='')

    plt.show()



if __name__ == "__main__":
    main()
