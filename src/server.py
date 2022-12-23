from flask import Flask, request
from main import QPlayer
from sheshbesh import to_str

app = Flask(__name__)

player = QPlayer(checkpoints_dir='.')
player.load('best_model-5-layers-750-latent-dim-False-doubles')

@app.route('/play', methods=['GET', 'POST'])
def hello():
    body = request.json
    state = body['state']
    dice = (body['roll'][0], body['roll'][1])

    print(to_str(state))
    print(dice)

    seq = player.play(state, dice)

    response = list(map(lambda x: dict(from_col=x[0][0], steps=x[0][1]), seq))

    print(f'moves --> {response}')

    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)