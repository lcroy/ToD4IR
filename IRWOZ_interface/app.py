from flask import Flask, render_template,request,jsonify
from flask_socketio import SocketIO
from utils.dbsearch import *
from config import Config

app = Flask(__name__)
app.config['SECRET_KEY'] = 'thisisforates###123'
socketio = SocketIO(app)
cfg = Config()

db_file = cfg.dataset_path_production_db


@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')


@app.route('/get_area/', methods=['GET'])
def get_area():
    area = request.args.get('area')
    if area == "":
        result = "null"
    else:
        result = query_area(db_file, area)

    return jsonify({"result":result})


@app.route('/get_location/', methods=['GET'])
def get_location():
    area = request.args.get('area')
    location = request.args.get('location')
    if (area == ""):
        if (location == ""):
            result = "Need Area and Location"
        else:
            result = "Need Area"
    else:
        if (location == ""):
            result = "Need Location"
        else:
            result = result = query_area_location(db_file, area, location)

    return jsonify({"result":result})


def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')


@socketio.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    print('received my event: ' + str(json))
    socketio.emit('my response', json, callback=messageReceived)


if __name__ == '__main__':
    socketio.run(app, debug=True)
