from main2 import Game, ResSequential, device, encode
from sheshbesh import end_turn, winner, to_str, can_step, step, single_step_actions, roll
import torch as t
import torch.nn.functional as F
import traceback
import inquirer


def human(game: Game):
    (x,y) = game.dice

    avail = [x,y]
    if x == y:
        avail *= 2
    
    s = game.state
    

def main():
    temperature = 0.01
    game = Game()

    Q = ResSequential()
    Q.load_latest()
    Q.to(device)

    while True:
        print(to_str(game.state))
        print(f'Dice {game.dice}')
        encodings = []
        # for game in games:
        for seq, _, _ in game.actions:
            for s,a,d in seq: 
                encodings.append(encode(s,a,d))
            
        if len(encodings) > 0:
            evals = Q.to(device)(t.stack(encodings).to(device)).flatten()
        else:
            evals = t.tensor([], device=device)

        idx = 0
        if len(game.actions) > 0:
            predicted_rewards = []
            for (seq, _, _) in game.actions:
                predicted_reward = t.tensor(0.0, device=device)
                for (s,a,d) in seq:
                    predicted_reward += evals[idx]
                    idx += 1
                predicted_rewards.append(predicted_reward)
            predicted_rewards = t.stack(predicted_rewards)

            # choose a sequence
            probs = F.softmax(predicted_rewards / temperature, dim=0)
            chosen_idx = t.distributions.categorical.Categorical(probs=probs).sample()
            print(f'I play: {[a for s,a,d in game.actions[chosen_idx][0]]}')
            game.choose(chosen_idx)
        else:
            print(f'I pass')
            game.pass_move()

        print('Your turn')

        win = winner(game.state)
        if win is not None:
            print('I WIN' if win else 'You win...')
            return

        human(game)
        
        win = winner(game.state)
        if win is not None:
            print('I WIN' if win else 'You win...')
            return

if __name__ == '__main__':
    main()