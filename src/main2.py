import torch as t
import os
from sheshbesh import flip_state, flip_action, house_is_full, initial_state, actions, roll, winner, end_turn, to_str, step, n_open_houses
from random import random, choices, randrange
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from tqdm import tqdm
import pickle

hyper_params = dict(
    n_layers=8,
    latent_dim=3000,
    max_examples=100000,
    batch_size=60000,
    time_limit=400,
    lr=1e-2,
    n_parallel_games=2,
)

t.set_default_dtype(t.float32)

device = t.device('cpu')
print(f'#CPUs: {os.cpu_count()}')
if t.cuda.is_available():
    device = t.device('cuda:0')
    info = t.cuda.get_device_properties(0)
    print(f'GPU: {info.name} (vram={info.total_memory}) (units={info.multi_processor_count})')
elif hasattr(t.backends, "mps") and t.backends.mps.is_available():
    device = t.device('mps')
    print(f'Using Apple Metal')
else:
    print('NO GPU')

class ResSequential(nn.Module):
    def __init__(self, n_layers=hyper_params['n_layers'], latent_dim=hyper_params['latent_dim'], out_dim=1, checkpoints_dir='./checkpoints'):
        super(ResSequential, self).__init__()
        self.layers = nn.ModuleList([nn.LazyLinear(latent_dim) for _ in range(n_layers)])
        self.final_layer = nn.LazyLinear(out_dim)
        self.checkpoints_dir = f'{checkpoints_dir}/n_layers={n_layers}-latent_dim={latent_dim}'

    def forward(self, data):
        data = F.relu(self.layers[0](data))
        for layer in self.layers[1:]:
            data = F.relu(layer(data) + data)
        return self.final_layer(data)

    def save(self, path):
        t.save(self.state_dict(), path)

    def get_checkpoints(self):
        files = os.listdir(self.checkpoints_dir)
        files.sort()
        return files

    def load(self, checkpoint):
        path = os.path.join(self.checkpoints_dir,checkpoint)
        self.load_state_dict(t.load(path))
        self.eval()

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
        self.save(os.path.join(self.checkpoints_dir, latest))
        print(f'Saved checkpoint: {latest}')

def encode(state, act, avail_dice):
    if not state[0]:
        state = flip_state(state)
        act = flip_action(act)

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
    
def returns(rews, discount=1-1e-4):
    
    discounts = discount**t.arange(len(rews))
    
    rews = t.tensor(rews)
    rews = rews.float()
    rews = rews * discount

    return t.cumsum(rews.flip(0), 0).flip(0) / discounts

def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-10)

class Memory:
    def __init__(self, max_examples=hyper_params['max_examples']):
        self.examples = []
        self.max_examples = max_examples

    def save(self):
        with open('memory', 'wb') as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open('memory', 'rb') as handle:
            self.examples = pickle.load(handle)

    def sample(self, n=hyper_params['batch_size']):
        batch = choices(self.examples, k=n)
        inputs = t.stack([input for input, _ in batch])
        targets = t.stack([target for _, target in batch])

        return inputs, targets

    def insert(self, input, target):
        self.examples.append((input, target))
        while len(self.examples) > self.max_examples:
            self.examples.pop(randrange(len(self.examples)))

    def __len__(self):
        return len(self.examples)

class Game:
    def __init__(self) -> None:
        self.turn = 1
        self.state = initial_state
        self.dice = roll()
        self.actions = list(actions(self.state, self.dice))
        self.white_episode = []
        self.black_episode = []

    def choose(self, idx):
        seq, rew, end_state = self.actions[idx]

        if self.state[0]:
            self.white_episode.append((seq, rew))
        else:
            self.black_episode.append((seq, rew))
        self.state = end_state
        self.end_turn()

    def play_moves(self, seq):
        rew = 0
        for move in seq:
            self.state, r = step(self.state, move)
            rew += r
        if self.state[0]:
            self.white_episode.append((seq, rew))
        else:
            self.black_episode.append((seq, rew))
        
        self.end_turn()

    def pass_move(self):
        self.end_turn()

    def end_turn(self):
        self.turn += 1
        self.state = end_turn(self.state)
        self.dice = roll()
        self.actions = list(actions(self.state, self.dice))

    def add_examples(self, memory: Memory):
        rets = returns([rew for _,rew in self.white_episode])
        for i, (seq, rew) in enumerate(self.white_episode):
            ret = rets[i]
            for (s,a,d) in seq:
                memory.insert(encode(s,a,d), ret / len(seq))
        
        rets = returns([rew for _,rew in self.black_episode])
        for i, (seq, rew) in enumerate(self.black_episode):
            ret = rets[i]
            for (s,a,d) in seq:
                memory.insert(encode(s,a,d), ret / len(seq))

def greedy_play(game: Game):
    # picks the sequence which maximizes the reward for this turn
    if len(game.actions):
        idx = max(enumerate(game.actions), key=lambda act: (act[1][1], random()))[0]
        game.choose(idx)
    else:
        game.pass_move()

def random_play(game: Game):
    # picks a random sequence
    if len(game.actions):
        idx = max(enumerate(game.actions), key=lambda _: random())[0]
        game.choose(idx)
    else:
        game.pass_move()

def safe_play(game: Game):
    # picks the sequence which results in the minimal number of open houses, if multiple exists, pick a the greedy option
    if len(game.actions):
        idx = max(enumerate(game.actions), key=lambda act: (-n_open_houses(act[1][2]), act[1][1], random()))[0]
        game.choose(idx)
    else:
        game.pass_move()

def play(Q: ResSequential, games, n_total_games=100, benchmark_player=None, temperature=1, on_game_end=None) -> float:
    opponent = benchmark_player.__name__ if benchmark_player else 'Self'
    print(f'Playing {n_total_games} games ({len(games)} in parallel) against {opponent} with temperature={temperature}')
    results = []
    
    with tqdm(total=n_total_games, unit='game', smoothing=0) as pbar:
        step = 0    
        while len(results) < n_total_games:
            pbar.set_description(f'Step #{step}')
            if (step % 2 == 0) or (benchmark_player is None):
                encodings = []
                for game in games:
                    for seq, _, _ in game.actions:
                        for s,a,d in seq: 
                            encodings.append(encode(s,a,d))
                    
                if len(encodings) > 0:
                    evals = Q(t.stack(encodings).to(device)).flatten()
                else:
                    evals = t.tensor([], device=device)

                idx = 0
                for (i, game) in enumerate(games):
                    if len(game.actions) > 0:
                        predicted_rewards = []
                        for (seq, _, _) in game.actions:
                            predicted_reward = t.tensor(0.0, device=device)
                            for (s,a,d) in seq:
                                predicted_reward += evals[idx]
                                idx += 1
                            predicted_rewards.append(predicted_reward)
                        predicted_rewards = t.stack(predicted_rewards).float()

                        # choose a sequence
                        probs = F.softmax(predicted_rewards / temperature, dim=0)
                        chosen_idx = t.distributions.categorical.Categorical(probs=probs).sample()
                        game.choose(chosen_idx)
                    else:
                        game.pass_move()
            elif benchmark_player is not None:
                for game in games:
                    benchmark_player(game)

            for i, game in enumerate(games):
                win = winner(game.state)
                if game.turn > hyper_params['time_limit'] or (win is not None):
                    results.append(win)
                    pbar.update(1)
                    if on_game_end is not None:
                        on_game_end(game, win)
                    games[i] = Game()
            
            step += 1

    results = list(filter(lambda x: x is not None, results))
    return 0 if len(results) == 0 else sum(results) / len(results)

def main():
    lr = hyper_params['lr']
    print(f'Learing rate = {lr}')

    Q = ResSequential()
    Q.load_latest()
    Q.to(device)
    loss_fn = nn.MSELoss()
    optim = t.optim.Adam(Q.parameters(), lr=lr)
    memory = Memory()

    def on_game_end(game, win):
        game.add_examples(memory)

    games = [
        [Game() for _ in range(hyper_params['n_parallel_games'])]
        for _ in range(4)
    ]

    while True:
        winrate = play(Q, games[0], n_total_games=hyper_params['n_parallel_games'], on_game_end=on_game_end, temperature=0.15)
        print(f'Winrate against self = {winrate}')
        winrate = play(Q, games[1], n_total_games=hyper_params['n_parallel_games'], on_game_end=on_game_end, temperature=0.01, benchmark_player=random_play)
        print(f'Winrate against random_player = {winrate}')
        winrate = play(Q, games[2], n_total_games=hyper_params['n_parallel_games'], on_game_end=on_game_end, temperature=0.01, benchmark_player=greedy_play)
        print(f'Winrate against greedy_player = {winrate}')
        winrate = play(Q, games[3], n_total_games=hyper_params['n_parallel_games'], on_game_end=on_game_end, temperature=0.01, benchmark_player=safe_play)
        print(f'Winrate against safe_player = {winrate}')

        inputs, targets = memory.sample()
        targets = normalize(targets)
        evals = Q(inputs.to(device)).flatten()
        loss = loss_fn(evals, targets.to(device))
        Q.zero_grad()
        loss.backward()
        optim.step()
        print(f'Loss={loss}')
        Q.save_checkpoint()

if __name__ == '__main__':
    main()