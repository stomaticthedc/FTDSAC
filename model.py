import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import RelaxedOneHotCategorical, Categorical
from torch.nn import TransformerEncoder, TransformerEncoderLayer

LOG_SIG_MAX = 2
LOG_SIG_MIN = -10
epsilon = 1e-6


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()
        #self.nhead = 5
        #transformer
        #encoder_layers = TransformerEncoderLayer(num_inputs, self.nhead)
        #self.transformer = TransformerEncoder(encoder_layers, num_layers=self.nhead)

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, num_actions)

        # Q2 architecture
        self.linear5 = nn.Linear(num_inputs, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, hidden_dim)
        self.linear7 = nn.Linear(hidden_dim, hidden_dim)
        self.linear8 = nn.Linear(hidden_dim, num_actions)

        # Q value combiner
        self.q_value_combiner = nn.Linear(num_actions, 1)  # 新添加的全连接层

        self.apply(weights_init_)

    def forward(self, state):
        #state = self.transformer(state)
        x1 = F.relu(self.linear1(state))
        x1 = F.relu(self.linear2(x1))
        x1 = F.relu(self.linear3(x1))
        x1 = self.linear4(x1)

        x2 = F.relu(self.linear5(state))
        x2 = F.relu(self.linear6(x2))
        x2 = F.relu(self.linear7(x2))
        x2 = self.linear8(x2)


        combined_q1 = self.q_value_combiner(x1)
        combined_q2 = self.q_value_combiner(x2)
        return combined_q1, combined_q2
        #return x1, x2

class ModifiedPolicy_noLSTM(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ModifiedPolicy_noLSTM, self).__init__()
        self.nhead = 5

        encoder_layers = TransformerEncoderLayer(num_inputs, self.nhead)
        self.transformer = TransformerEncoder(encoder_layers, num_layers=self.nhead)

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.content_index_linear = nn.Linear(hidden_dim, 50)  # 输出内容索引
        self.bandwidth_allocation_linear = nn.Linear(hidden_dim, 30)  # 输出带宽分配

        self.apply(weights_init_)

    def forward(self, state):

        state = self.transformer(state)

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        content_index_probs = F.softmax(self.content_index_linear(x), dim=1)
        bandwidth_allocation_probs = F.softmax(self.bandwidth_allocation_linear(x).view(-1, 10, 3), dim=-1)

        return content_index_probs, bandwidth_allocation_probs

    def sample(self, state):
        content_index_probs, bandwidth_allocation_probs = self.forward(state)
        batch_size = bandwidth_allocation_probs.shape[0]
        bandwidth_allocation_probabilities_flat = bandwidth_allocation_probs.view(batch_size, -1)

        action_prob = torch.cat([content_index_probs, bandwidth_allocation_probabilities_flat],  dim=1)
        # 对内容索引采样
        content_index_distribution = Categorical(content_index_probs + epsilon)
        content_index = content_index_distribution.sample()
        log_content_index_probabilities = torch.log(content_index_probs)

        # 对带宽分配采样
        bandwidth_distribution = Categorical(bandwidth_allocation_probs + epsilon)
        bandwidth_allocation = bandwidth_distribution.sample()
        log_bandwidth_allocation_probabilities = torch.log(bandwidth_allocation_probs)

        # 为了避免设备不匹配，我们需要确定所有的张量都在同一设备上
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 转移张量到目标设备
        content_index = content_index.to(device)
        bandwidth_allocation = bandwidth_allocation.to(device)
        log_bandwidth_allocation_probabilities = log_bandwidth_allocation_probabilities.to(device)

        # 对内容索引和带宽分配进行拼接
        # 获取 content_index 和 bandwidth_allocation 的尺寸
        content_index_shape = content_index.shape
        bandwidth_allocation_shape = bandwidth_allocation.shape

        # 重塑 content_index
        if len(content_index_shape) == 1:
            # 如果 content_index 是一维的，增加一个尺寸
            content_index = content_index.view(content_index_shape[0], 1)

        # 现在 content_index 是二维的，我们可以用 cat 操作将它与 bandwidth_allocation 拼接在一起
        action = torch.cat([content_index, bandwidth_allocation], dim=1)

        # 如果原始的 content_index 是一维的，我们需要去掉增加的尺寸
        if len(content_index_shape) == 1:
            action = action.squeeze(0)

            # 对log概率进行拼接
        log_bandwidth_allocation_probabilities_flat = log_bandwidth_allocation_probabilities.view(batch_size, -1)

        log_action_probabilities = torch.cat([log_content_index_probabilities, log_bandwidth_allocation_probabilities_flat], dim=1)

        return action, (action_prob, log_action_probabilities)

    def to(self, device):
        return super(ModifiedPolicy_noLSTM, self).to(device)

# original policy network
class GaussianPolicy_orig(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy_orig, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)
        #nn.Transformer()
        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy_orig, self).to(device)


# policy with lstm
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state, hidden_in):
        """
        :param state: 3-d for lstm
        :param hidden_in: hidden state of lstm
        :return:
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))

        x = torch.unsqueeze(x, dim=0)  # more dimension
        x, hidden_lstm = self.lstm(x, hidden_in)
        x = torch.squeeze(x, dim=0)  # less dimension

        probability = F.softmax(self.mean_linear(x), dim=1)
        return probability, hidden_lstm

    def sample(self, state, hidden_in):
        action_probabilities, hidden_lstm = self.forward(state, hidden_in)
        action_distribution = Categorical(action_probabilities)
        action = action_distribution.sample()
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), hidden_lstm

    def to(self, device):
        return super(GaussianPolicy, self).to(device)
