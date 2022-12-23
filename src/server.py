from flask import Flask, request
from main import QPlayer

app = Flask(__name__)

player = QPlayer(checkpoints_dir='.')
player.load('best_model-5-layers-750-latent-dim-False-doubles')

@app.route('/')
def hello():
    body = request.json
    print(body)
    state = body['state']
    dice = (body['roll'][0], body['roll'][1])

    seq = player.play(state, dice)

    return list(map(lambda x: dict(from_col=x[0][0], steps=x[0][1]), seq))