from flask import Flask, render_template,request,jsonify
from flask_socketio import SocketIO
from utils.dbsearch import *
from config import Config
from datetime import datetime
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'thisisforates###123'
socketio = SocketIO(app)
cfg = Config()

start_conv = 1
domain_set = ''
user_utterance = ''
turn = []
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


@app.route('/get_end_conv/', methods=['GET'])
def get_end_conv():
    global start_conv
    global domain_set
    global turn
    end_conv = request.args.get('end')
    dialogue = {}
    print("END")
    if end_conv == "yes":
        # format data
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H-%M-%S")
        dialogue_name = timestampStr + ".json"
        dialogue[dialogue_name] = {}
        dialogue[dialogue_name]["domain"] = domain_set
        dialogue[dialogue_name]["turn"] = turn
        print(dialogue)
        # write to json file
        filename = cfg.dataset_path_gen_IRWOZ
        # 1. Read file contents
        with open(filename, "r") as file:
            data = json.load(file)
        # 2. Update json object
            data[dialogue_name] = dialogue[dialogue_name]
        # 3. Write json file
        with open(filename, "w") as file:
            json.dump(data, file)

        start_conv = 1

    return jsonify({"result":"done"})


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

@app.route('/get_prod/', methods=['GET'])
def get_prod():
    prod = request.args.get('prod')
    if prod == "":
        result = "null"
    else:
        result,_ = query_product(db_file, prod)

    return jsonify({"result":result})


@app.route('/get_position_name/', methods=['GET'])
def get_position_name():
    position_name = request.args.get('position_name')
    operation = request.args.get('operation')
    result = query_position(db_file, position_name, operation)

    return jsonify({"result":result})


@app.route('/get_object/', methods=['GET'])
def get_object():
    object_name = request.args.get('object')
    result = query_object(db_file, object_name)

    return jsonify({"result":result})

def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')


@socketio.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    global domain_set
    global start_conv
    global turn
    global user_utterance
    print('received my event: ' + str(json))
    raw_dialogue = json
    if json['flag'] == 'response':
        res = json['t_res'] + " " + json['s_res']
        json = {'speaker':json['speaker'],'message':res}

        # generate IRWOZ corpus from dialogue
        if raw_dialogue['speaker'] == 'max':
            if start_conv == 1:
                domain_set = raw_dialogue['domain_set']
                start_conv = 0
            raw_dialogue['slots']['user'] = user_utterance
            turn.append(raw_dialogue['slots'])
            print(turn)
        if raw_dialogue['speaker'] == 'user':
            user_utterance = raw_dialogue['slots']['user']

    socketio.emit('my response', json, callback=messageReceived)


if __name__ == '__main__':
    socketio.run(app, debug=True)
