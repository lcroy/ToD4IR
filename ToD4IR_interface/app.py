from flask import Flask, render_template,request,jsonify
from flask_socketio import SocketIO
from utils.dbsearch import *
from config import Config
from datetime import datetime
import json
import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from utils.dbsearch import *
from utils.normalize_text import normalize

app = Flask(__name__)
app.config['SECRET_KEY'] = 'thisisforates###123'
socketio = SocketIO(app)
cfg = Config()

# load model
model = GPT2LMHeadModel.from_pretrained(cfg.model_gpt2_checkpoint_path)
# get tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_gpt2_checkpoint_path)

model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

endoftext = tokenizer.encode(str(tokenizer._eos_token))
dialogue = '<|endoftext|> <|boc|> '
db_file = cfg.dataset_path_production_db

bf = ''
sys = ''
tt = ''
st = ''

@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')


def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')


# def db_search(context_pred_belief):
#
#     loc_db_req = context_pred_belief.find('<|DB_req|>')
#     loc_db_opt = context_pred_belief.find('<|DB_opt|>')
#     loc_t_req = context_pred_belief.find('<|T_req|>')
#     loc_t_opt = context_pred_belief.find('<|T_opt|>')
#     loc_eob = context_pred_belief.find('<|eob|>')
#     # get db_req slots
#     if loc_db_req > 0:
#         if loc_db_opt > 0:
#             db_req = context_pred_belief[loc_db_req + 10:loc_db_opt]
#         elif loc_t_req > 0:
#             db_req = context_pred_belief[loc_db_req + 10:loc_t_req]
#         elif loc_t_opt > 0:
#             db_req = context_pred_belief[loc_db_req + 10:loc_t_opt]
#         else:
#             db_req = context_pred_belief[loc_db_req + 10:loc_eob]
#
#         db_req = [x for x in list(dict.fromkeys(db_req.split(' '))) if x]
#
#     # get t_req slots
#     if loc_t_req > 0:
#         if loc_t_opt > 0:
#             t_req = context_pred_belief[loc_t_req + 9:loc_t_opt]
#         else:
#             t_req = context_pred_belief[loc_t_req + 9:loc_eob]
#
#         t_req = [x for x in list(dict.fromkeys(t_req.split(' '))) if x]
#
#     results = ''
#     # db_req slots
#     if loc_db_req > 0:
#         # get the domain
#         domain = db_req[0]
#         if domain == 'delivery':
#             # set the condition for searching area_location table
#             area, location = '', ''
#             for item in db_req[1:]:
#                 key, value = item[0:item.find('=')], item[item.find('=') + 1:]
#                 if key == 'area':
#                     area = value
#                 elif key == 'location':
#                     location = value
#             if (area == 'not_mentioned') and (location == 'not_mentioned'):
#                 results = 'area=null location=null'
#             elif (area != 'not_mentioned') and (location == 'not_mentioned'):
#                 db_results = query_area(cfg.dataset_path_production_db, area)
#                 results = 'area=' + db_results + ' location=null'
#             elif (area != 'not_mentioned') and (location != 'not_mentioned'):
#                 db_results = query_area_location(cfg.dataset_path_production_db, area, location)
#                 results = 'area=' + db_results + ' location=' + db_results
#             elif (area == 'not_mentioned') and (location != 'not_mentioned'):
#                 db_results = query_location(cfg.dataset_path_production_db, location)
#                 results = 'area=null' + ' location=' + db_results
#
#         elif domain == 'assembly':
#             # set the condition for searching product table
#             producttype = ''
#             for item in db_req[1:]:
#                 key, value = item[0:item.find('=')], item[item.find('=') + 1:]
#                 if key == 'producttype':
#                     producttype = value
#             if (producttype == 'not_mentioned'):
#                 results = 'producttype=null'
#             else:
#                 db_results, _ = query_product(cfg.dataset_path_production_db, producttype)
#                 results = 'producttype=' + db_results
#
#         elif domain == 'position':
#             # set the condition for searching position table
#             position_name = ''
#             for item in db_req[1:]:
#                 key, value = item[0:item.find('=')], item[item.find('=') + 1:]
#                 if key == 'position_name':
#                     producttype = value
#             if (position_name == 'not_mentioned'):
#                 results = 'position_name=null'
#             else:
#                 db_results, _ = query_position_name(cfg.dataset_path_production_db, position_name)
#                 results = 'position_name=' + db_results
#
#         elif domain == 'relocation':
#             # set the condition for searching position table
#             object_name = ''
#             for item in db_req[1:]:
#                 key, value = item[0:item.find('=')], item[item.find('=') + 1:]
#                 if key == 'object_name':
#                     object_name = value
#             if (object_name == 'not_mentioned'):
#                 results = 'object_name=null'
#             else:
#                 db_results, _ = query_object(cfg.dataset_path_production_db, object_name)
#                 results = 'object_name=' + db_results
#
#     # t_req slots
#     if loc_t_req > 0:
#         # get the domain
#         domain = t_req[0]
#         if domain == 'delivery':
#             # set the condition for searching area_location table
#             object = ''
#             for item in t_req[1:]:
#                 key, value = item[0:item.find('=')], item[item.find('=') + 1:]
#                 if key == 'object':
#                     object = value
#             if (object == 'not_mentioned'):
#                 results += ' object=null'
#             else:
#                 results += ' object=detected'
#
#         elif domain == 'assembly':
#             # set the condition for searching product table
#             product, quantity = '', ''
#             for item in t_req[1:]:
#                 key, value = item[0:item.find('=')], item[item.find('=') + 1:]
#                 if key == 'product':
#                     product = value
#                 elif key == 'quantity':
#                     quantity = value
#             if (product == 'not_mentioned'):
#                 results += ' product=null'
#             else:
#                 results += ' product=detected'
#             if (quantity == 'not_mentioned'):
#                 results += ' quantity=null'
#             else:
#                 results += ' quantity=detected'
#
#         elif domain == 'position':
#             operation = ''
#             for item in t_req[1:]:
#                 key, value = item[0:item.find('=')], item[item.find('=') + 1:]
#                 if key == 'operation':
#                     operation = value
#             if (operation == 'not_mentioned'):
#                 results += ' operation=null'
#             else:
#                 results += ' operation=detected'
#
#     return results

def do_generation(text):

        global dialogue
        global bf
        global sys
        global tt
        global st
        response = ''

        if len(text) > 0:
            text = normalize(text)
            # end of the current dialogue and go to next one
            if any(key in text for key in cfg.stop_words):
                dialogue = '<|endoftext|> <|boc|> '
                # text_to_speech_microsoft(random.choice(cfg.max_end_dialogue))
                return 'Please give next command...','','','',''

            # add the user utterance to the previous context the "<|eoc|>" location
            loc_eoc = dialogue.find("<|eoc|>")
            if loc_eoc < 0:
                # the beginning of the dialogue
                dialogue += '<|user|> ' + text + " <|eoc|>"
            else:
                # during the conversation
                tmp_context = dialogue.split('<|eoc|>', 1)
                dialogue = tmp_context[0] + ' <|user|> ' + text + tmp_context[1]

            # encoding the dialogue
            indexed_tokens = tokenizer.encode(dialogue)
            if len(indexed_tokens) > cfg.max_length:
                indexed_tokens = indexed_tokens[-1 * cfg.max_length:]
            tokens_tensor = torch.tensor([indexed_tokens])
            tokens_tensor = tokens_tensor.to(device)
            predicted_index = indexed_tokens[-1]

            # Task 1: generate belief
            with torch.no_grad():
                while predicted_index not in endoftext:
                    outputs = model(tokens_tensor)
                    predictions = outputs[0]
                    predicted_index = torch.argmax(predictions[0, -1, :]).item()
                    indexed_tokens += [predicted_index]

                    # monitoring the generated text
                    tmp = tokenizer.decode(indexed_tokens)

                    tokens_tensor = torch.tensor([indexed_tokens]).to(device)
                    # if it reach the max length, exit
                    if len(indexed_tokens) > cfg.max_length:
                        break
                    # if it reach the end of belief, exit
                    if tokenizer.decode(indexed_tokens).endswith('<|eob|>'):
                        break

            dialogue = tokenizer.decode(indexed_tokens)
            bf = dialogue[dialogue.find('<|bob|>') + 8: len(dialogue) - 7]
            print("belief================================")
            print(dialogue)
            print("belief================================")

            # Task 2.a: extract the domain and slots to query the DB
            sys_act = db_search(cfg, dialogue)

            # Task 2.b: generate system act (context + pred_belief + system actions)
            dialogue += ' <|bosys_act|> ' + sys_act + ' <|eosys_act|>'
            sys = sys_act

            print("system act================================")
            print(dialogue)
            print("system act================================")

            # Task 3: generation response
            indexed_tokens = tokenizer.encode(dialogue)
            if len(indexed_tokens) > cfg.max_length:
                indexed_tokens = indexed_tokens[-1 * cfg.max_length:]

            # Convert indexed tokens in a PyTorch tensor
            tokens_tensor = torch.tensor([indexed_tokens])

            tokens_tensor = tokens_tensor.to(device)
            predicted_index = indexed_tokens[-1]

            # Predict system response + small talk response
            with torch.no_grad():
                # while predicted_index not in endoftext:
                while predicted_index not in endoftext:
                    outputs = model(tokens_tensor)
                    predictions = outputs[0]
                    predicted_index = torch.argmax(predictions[0, -1, :]).item()
                    indexed_tokens += [predicted_index]

                    # monitoring the generated text
                    tmp = tokenizer.decode(indexed_tokens)

                    tokens_tensor = torch.tensor([indexed_tokens]).to(device)

                    # if it reach the max length, exit
                    if len(indexed_tokens) > cfg.max_length:
                        break
                    # if it reach the end of belief, exit
                    if tokenizer.decode(indexed_tokens).endswith('<|eoSres|>'):
                        break

                # get the response
                dialogue = tokenizer.decode(indexed_tokens)

                print("response================================")
                print(dialogue)
                print("response================================")


                # lex system response (sysTres)
                while dialogue.find('[') > 0:
                    loc_sysTres_slot_start = dialogue.find('[')
                    loc_sysTres_slot_end = dialogue.find(']')
                    slot_key = dialogue[loc_sysTres_slot_start + 1: loc_sysTres_slot_end] + "="
                    loc_slot_key = dialogue.find(slot_key)
                    temp = dialogue[loc_slot_key:]
                    slot_value = temp[temp.find('=') + 1: temp.find(' ')]
                    new_sentence = dialogue[:loc_sysTres_slot_start] + slot_value + dialogue[loc_sysTres_slot_end + 1:]
                    dialogue = new_sentence

                # extract the location of the sysTres and sysSres
                loc_boTres = dialogue.find('<|boTres|>')
                loc_eoTres = dialogue.find('<|eoTres|>')
                loc_boSres = dialogue.find('<|boSres|>')
                loc_eoSres = dialogue.find('<|eoSres|>')

                # just in case the response is not generated correctly
                response = "I do not understand. Can you repeat? Thanks."

                # extract the context
                loc_eoc = dialogue.find('<|eoc|>')
                new_dialogue = dialogue[:loc_eoc]
                # extract the the sysTres and sysSres
                if loc_boTres > 0:
                    t_res = dialogue[loc_boTres + 10: loc_eoTres]
                    r_res = dialogue[loc_boSres + 10: loc_eoSres]
                    new_dialogue += ' <|sys|> ' + t_res + " " + r_res
                    response = t_res + " " + r_res
                    tt = t_res
                    st = r_res

                dialogue = new_dialogue + ' <|eoc|>'
                # text_to_speech_microsoft(response)
                print("Max: " + response)

            return response, bf, sys, tt, st

@socketio.on('my event')
def handle_my_custom_event(json, methods=['GET', 'POST']):
    global dialogue
    print('received my event: ' + str(json))
    raw_dialogue = json
    if json['flag'] == 'response':
        res = json['utterance']
        json = {'speaker':json['speaker'],'message':res}

        # generate response
        response, bf, sys, tt, st = do_generation(res)
        json['tod_res'] = response
        json['bf'] = bf
        json['sys'] = sys
        json['tt'] = tt
        json['st'] = st

    socketio.emit('my response', json, callback=messageReceived)


if __name__ == '__main__':
    socketio.run(app, debug=True)
