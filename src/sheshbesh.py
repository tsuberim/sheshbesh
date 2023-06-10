from random import randrange
from numpy import sign

initial_state = [
    True,  # is white
    0,  # number of black eaten
    -2, 0, 0, 0, 0, 5,
    0, 3, 0, 0, 0, -5,
    5, 0, 0, 0, -3, 0,
    -5, 0, 0, 0, 0, 2,
    0   # number of white eaten
]

def flip_state(state):
    [is_white, *board] = state
    parts = (board[1:1+6], board[1+6:1+2*6], board[1+2*6:1+3*6], board[1+3*6:1+4*6])
    new_board = [board[-1], *parts[3], *parts[2], *parts[1], *parts[0], board[0]]
    new_board = board[::-1]
    return [not(is_white), *map(lambda x: -x, new_board)]

def flip_action(act):
    (from_col, steps) = act
    return (25-from_col, steps)

def flip_seq(seq):
    if seq is None:
        return None

    new_seq = []
    for (act, avail_dice) in seq:
        new_seq.append((flip_action(act), avail_dice))
    return new_seq

def flip_episode(episode):
    return [(flip_state(state), flip_seq(seq), rew) for (state, seq, rew) in episode]

def require(cond, msg):
    if not (cond):
        raise RuntimeError(msg)


def winner(state):
    [is_white, *board] = state
    has_whites = sum(n > 0 for n in board)
    has_blacks = sum(n < 0 for n in board)
    if has_whites == 0:
        return True
    if has_blacks == 0:
        return False
    return None


def house_is_full(state, is_white=True):
    [_, *board] = state
    return sum(n > 0 if is_white else n < 0 for n in (board[7:] if is_white else board[:19])) == 0


def step(state, act):
    [is_white, *board] = state
    (from_col, steps) = act

    rew = -7

    s = 1 if is_white else -1

    eaten_col = 25 if is_white else 0
    has_eaten = abs(board[eaten_col]) > 0

    require(has_eaten == (from_col == eaten_col),
            'You are eaten, must enter opposite color house')
    require(sign(board[from_col]) == s, 'You have to move your own pieces')

    board[from_col] -= s
    to_col = from_col - s*steps

    if 1 <= to_col and to_col < len(board) - 1:
        require(board[to_col] > -2 if is_white else board[to_col]
                < 2, 'Cannot move to opposite color house')
        if board[to_col] == -s:
            board[0 if is_white else 25] -= s
            board[to_col] = 0
            rew += to_col if is_white else len(board) - to_col
        board[to_col] += s
        rew += steps
    else:
        require(house_is_full(state, is_white),
                'Cannot take out pieces when house is not full')
        rew += 24

    state = [is_white, *board]
    win = winner(state)
    if win is not None:
        rew += 100 if (win == is_white) else -100

    return state, rew


def can_step(state, act):
    try:
        step(state, act)
        return True
    except:
        return False


def end_turn(state):
    [is_white, *board] = state
    return [not (is_white), *board]


def single_step_actions(state, steps):
    [is_white, *board] = state
    return ((from_col, steps) for from_col in range(len(board)) if can_step(state, (from_col, steps)))


from typing import Tuple, Generator, List
State = object
Action = Tuple[int, int]
ActSeq = Tuple[State, Action, List[int]]
Something = Tuple[ActSeq, float, State]

def actions(state, dice, doubles=True) -> Generator[Something, None, None]:
    (x, y) = dice

    if x == y and doubles:
        for a1 in single_step_actions(state, x):
            s1, r1 = step(state, a1)
            if winner(s1) is not None:
                yield ((state, a1, (x, x, x,)),), r1, s1
            else:
                for a2 in single_step_actions(s1, x):
                    s2, r2 = step(s1, a2)
                    if winner(s1) is not None:
                        yield ((state, a1, (x, x, x)), (s1, a2, (x, x,))), r1 + r2, s2
                    else:
                        for a3 in single_step_actions(s2, x):
                            s3, r3 = step(s2, a3)
                            if winner(s1) is not None:
                                yield ((state, a1, (x, x, x)), (s1, a2, (x, x)), (s2, a3, (x,))), r1 + r2 + r3, s3
                            else:
                                for a4 in single_step_actions(s3, x):
                                    s4, r4 = step(s3, a4)
                                    yield ((state, a1, (x, x, x)), (s1, a2, (x, x)), (s2, a3, (x,)), (s3, a4, ())), r1 + r2 + r3 + r4, s4

    else:
        for act1 in single_step_actions(state, x):
            s1, r1 = step(state, act1)
            if winner(s1) is not None:
                yield ((state, act1, (y,)),), r1, s1
            else:
                for act2 in single_step_actions(s1, y):
                    s2, r2 = step(s1, act2)
                    yield ((state, act1, (y,)), (s1, act2, ())), r1 + r2, s2

        for act1 in single_step_actions(state, y):
            s1, r1 = step(state, act1)
            if winner(s1) is not None:
                yield ((state, act1, (x,)),), r1, s1
            else:
                for act2 in single_step_actions(s1, x):
                    s2, r2 = step(s1, act2)
                    yield ((state, act1, (x,)), (s1, act2, ())), r1 + r2, s2


def to_str(state):
    [is_white, *board] = state

    char = 'oo' if is_white else '**'

    def col_to_str(n):
        char = '*' if n < 0 else 'o'
        return (char * abs(n)) + ('-' * (15 - abs(n)))

    out = ''
    out += '=' * 37 + '\n'
    for col in range(6):
        out += (str(1 + col).zfill(2)) + '|' + col_to_str(board[1 + col]) + '|' + \
            col_to_str(board[::-1][1 + col])[::-1] + \
            '|' + (str(24-col).zfill(2)) + '\n'
    out += '=' * 37 + '\n'
    out += char + '|' + col_to_str(board[0])[::-1] + \
        '|' + col_to_str(board[25]) + '|' + char + '\n'
    out += '=' * 37 + '\n'
    for col in range(6):
        out += (str(7 + col).zfill(2)) + '|' + col_to_str(board[7 + col]) + '|' + \
            col_to_str(board[::-1][7 + col])[::-1] + \
            '|' + (str(18-col).zfill(2)) + '\n'
    out += '=' * 37 + '\n'
    return out


def roll():
    return (randrange(1, 6), randrange(1, 6))

def n_open_houses(state):
    [is_white, *board] = state
    s = 1 if is_white else -1
    return sum([i == s for i in board[1:-1]])