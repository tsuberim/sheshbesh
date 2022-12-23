from random import random, randrange, choice, choices
import torch as t
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
from sheshbesh import initial_state, actions, can_step, step, end_turn, single_step_actions, to_str, winner, roll, house_is_full, flip_episode, flip_state, flip_seq
from timeit import timeit
import ray
from itertools import cycle
from datetime import datetime
import os
import numpy as np

device = t.device("cuda:1" if t.cuda.is_available() else "cpu")

hyper_params = dict(
    n_layers=5,
    latent_dim=1100,
    dropout=0.3,
    doubles=True,
    discount=1-1e-4,
    max_eps=7000,
    flip=False
)

experiment = '-'.join([f'{k}={v}' for (k,v) in hyper_params.items()])

class ResSequential(nn.Module):
    def __init__(self, n_layers=hyper_params['n_layers'], latent_dim=hyper_params['latent_dim'], out_dim=1, dropout=hyper_params['dropout']):
        super(ResSequential, self).__init__()
        self.layers = nn.ModuleList([nn.LazyLinear(latent_dim) for _ in range(n_layers)])
        self.final_layer = nn.LazyLinear(out_dim)
        self.dropout = dropout

    def forward(self, data):
        data = F.elu(F.dropout(self.layers[0](data), p=self.dropout))
        for layer in self.layers[1:]:
            data = F.elu(F.dropout(layer(data) + data, p=self.dropout))
        return self.final_layer(data)

    def save(self, path):
        t.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(t.load(path))
        self.eval()


class QPlayer:
    def __init__(self, lr: float = 7e-5, checkpoints_dir=f'../checkpoints-{experiment}'):
        self.Q = ResSequential()
        self.loss_fn = nn.MSELoss()
        self.optim = t.optim.Adam(self.Q.parameters(), lr=lr)
        self.checkpoints_dir = checkpoints_dir

    def get_checkpoints(self):
        files = os.listdir(self.checkpoints_dir)
        files.sort()
        return files

    def load(self, checkpoint):
        self.Q.load(os.path.join(self.checkpoints_dir,checkpoint))

    def load_latest(self):
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        files = self.get_checkpoints()
        if len(files):
            latest = files[-1]
            self.load(latest)
            print(f'Loaded latest checkpoint: {latest}')

    def save_checkpoint(self):
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        
        time = datetime.now().isoformat().replace(':', '-').replace('.', '-')
        latest = f'checkpoint-{time}'
        self.Q.save(os.path.join(self.checkpoints_dir, latest))
        print(f'Saved checkpoint: {latest}')

    def encode(self, state, act, avail_dice):
        [is_white, *board] = state
        (from_col, steps) = act

        # encode board
        e = t.eye(31)
        vecs = [e[n] for n in board]

        # encode action
        vecs.append(t.eye(26)[from_col])
        vecs.append(t.eye(6)[steps])

        # encode helper indicators
        indicators = [
            is_white,
            house_is_full(state, is_white),
            house_is_full(state, not (is_white)),
            house_is_full(state, True),
            house_is_full(state, False),
        ]
        for ind in indicators:
            vecs.append(t.tensor([1.0 if ind else -1.0]))

        # encode available dice
        avail_dice_vec = t.zeros(6)
        for d in avail_dice:
            avail_dice_vec[d] += 1
        vecs.append(avail_dice_vec)

        return t.concat(vecs)

    def evaluate_seqs(self, states, seqs, cuda=False):
        state_acts = []
        for (state, (seq, rew)) in zip(cycle(states), seqs):
            s = state
            if seq is not None:
                for (act, avail_dice) in seq:
                    state_acts.append(self.encode(s, act, avail_dice))
                    s, _ = step(s, act)

        state_act_evals = self.Q(t.stack(state_acts).cuda() if cuda else t.stack(state_acts))

        seq_evals = []
        i = 0
        for (seq, rew) in seqs:
            total = t.tensor([0.0])
            if cuda: total = total.cuda()
            if seq is not None:
                for (act, avail_dice) in seq:
                    total += state_act_evals[i]
                    i += 1
            seq_evals.append((seq, total))
        return seq_evals

    def play(self, state, dice):
        is_white = state[0]
        if not is_white:
            state = flip_state(state)

        seqs = list(actions(state, dice, doubles=hyper_params['doubles']))

        if len(seqs) == 0:
            return None

        seq_evals = self.evaluate_seqs([state], seqs)
        seq = max(seq_evals, key=lambda seq: (seq[1], random()))[0]

        if not is_white:
            seq = flip_seq(seq)
        return seq

    def loss(self, episodes, cuda=False):
        rets = t.concat([returns([rew for (state, act, rew) in episode]) for episode in episodes])
        if cuda: rets = rets.cuda()

        states = sum([[state for (state, seq, rew) in episode] for episode in episodes], [])
        seqs = sum([[(seq, rew) for (state, seq, rew) in episode] for episode in episodes], [])

        seq_evals = t.concat([seq_eval for (seq, seq_eval) in self.evaluate_seqs(states, seqs, cuda=cuda)])

        loss = self.loss_fn(seq_evals, rets)
        return loss

    def learn(self, episodes, cuda=False):
        if cuda: self.Q.cuda()
        loss = self.loss(episodes, cuda=cuda)
        self.Q.zero_grad()
        loss.backward()
        self.optim.step()
        self.Q.cpu()
        return loss


def returns(rews, discount=hyper_params['discount']):
    discounts = discount**t.arange(len(rews))
    rews = t.tensor(rews) * discounts
    return t.cumsum(rews.flip(0), 0).flip(0) / discounts


class Memory:
    def __init__(self, max_eps=hyper_params['max_eps']):
        self.eps = []
        self.max_eps = max_eps

    def sample(self, n=100):
        return choices(self.eps, k=n)

    def insert(self, ep):
        self.eps.append(ep)
        while len(self.eps) > self.max_eps:
            self.eps.pop(randrange(len(self.eps)))

    def __len__(self):
        return len(self.eps)


class RandomPlayer:
    def play(self, state, dice):
        seqs = list(actions(state, dice, doubles=hyper_params['doubles']))
        if len(seqs) == 0:
            return None

        return choice(seqs)[0]


class GreedyPlayer:
    def play(self, state, dice):
        seqs = list(actions(state, dice, doubles=hyper_params['doubles']))
        if len(seqs) == 0:
            return None
        return max(seqs, key=lambda seq: (seq[1], random()))[0]

class MixPlayer:
    def __init__(self, p1, p2, alpha=0.5):
        self.p1 = p1
        self.p2 = p2
        self.alpha = alpha

    def play(self, state, dice):
        if random() <= self.alpha:
            return self.p1.play(state, dice)
        else:
            return self.p2.play(state, dice)

@ray.remote
def play(p1, p2, render=False):
    players = (p1, p2)
    episodes = ([], [])
    state = initial_state
    turns = 0
    while winner(state) is None:
        if render:
            print(to_str(state))
        dice = roll()
        if render:
            print(f'dice={dice}')
        last_state = state
        seq = players[turns % 2].play(state, dice)
        total_rew = 0
        if seq is not None:
            if render:
                print(f'seq={seq}')
            for (act, avail_dice) in seq:
                state, rew = step(state, act)
                total_rew += rew
                if render:
                    print(f'rew={rew}')
        else:
            if render:
                print('No moves - PASS')
        state = end_turn(state)
        episodes[turns % 2].append((last_state, seq, rew))
        turns += 1

    return winner(state), episodes, turns


def winrate(p1, p2, n=100):
    episodes = ([], [])

    def playrun():
        flip = random() < .5 if hyper_params['flip'] else False
        result= play.remote(p2, p1, render=False) if flip else play.remote(p1, p2, render=False)
        return result, flip 

    def handle(ref, flip):
        white_wins, (ep1, ep2), _ = ray.get(ref)
        episodes[0 ^ flip].append(ep1)
        episodes[1 ^ flip].append(ep2)
        if white_wins is None:
            return None

        return white_wins ^ flip

    runs = [playrun() for _ in range(n)]
    results = [handle(ref, flip) for (ref, flip) in runs]

    return (sum(filter(lambda x: x is not None, results)) / n, *episodes)

def main():
    # with open('src/requirements.txt', 'r', encoding='utf-8') as f:
    #     requirements = f.readlines()

    # ray.init(address='ray://127.0.0.1:10001', runtime_env=dict(
    #     working_dir='src',
    #     pip=['torch', 'numpy', 'tensorboard']
    # ))
    ray.init()
    writer = SummaryWriter(comment=experiment)

    n_games = 30
    n_players = 2
    n_checkpoints = 1
    n_learnings_per_batch = 3

    episodes_per_batch = 2*n_games*(n_players + n_checkpoints)

    mem = Memory(4*episodes_per_batch)

    random_player = RandomPlayer()
    greedy_player = GreedyPlayer()
    q_player = QPlayer()
    q_player.load_latest()

    def play_against(player, name, n):
        wr, eps1, eps2 = winrate(q_player, player, n=n)
        writer.add_scalar(f'{experiment}/vs. {name}', wr, i)
        print(f'{datetime.now().isoformat()} {i}) vs. {name} = {wr}')
        for ep in eps1: mem.insert(ep)
        for ep in eps2: mem.insert(flip_episode(ep))

    try:
        i = 0
        while True:
            for alpha in np.linspace(.6,.8,n_players):
                play_against(MixPlayer(greedy_player, random_player, alpha), f'{alpha*100}% greedy', n_games)

            for checkpoint in q_player.get_checkpoints()[::-1][:n_checkpoints]:
                checkpoint_player = QPlayer()
                checkpoint_player.load(checkpoint)
                play_against(checkpoint_player, checkpoint, n_games)

            for _ in range(n_learnings_per_batch):
                loss = q_player.learn(mem.sample(episodes_per_batch))
                writer.add_scalar(f'{experiment}/loss', loss, i)
                print(f'{datetime.now().isoformat()} {i}) loss = {loss}')

            if i > 0 and i % 5 == 0:
                q_player.save_checkpoint()
            i += 1
    except KeyboardInterrupt:
        q_player.save_checkpoint()
        os._exit(0)

if __name__ == '__main__':
    print(timeit(main, number=1))
