import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from config import Config
# from utils.speech_service import *
from utils.dbsearch import *
from utils.normalize_text import normalize
import logging
from torch.utils.data import DataLoader, Dataset

cfg = Config()
logger = logging.getLogger(__name__)

def decoder(lines):
    model = GPT2LMHeadModel.from_pretrained(cfg.model_gpt2_checkpoint_path)
    # get tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_gpt2_checkpoint_path)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    endoftext = tokenizer.encode(str(tokenizer._eos_token))
    dec_dialogue = []

    for line in lines:
        loc_end_of_contx = line.find('<|eoc|>')
        dialogue = line[:loc_end_of_contx+ 7]
        print("Before decoding ================")
        print(dialogue)
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
        # print the intermediate results
        # print(dialogue)
        # text_to_speech_microsoft(context_pred_belief)
        # print("belief================================")
        # print(dialogue)
        # print("belief================================")

        # Task 2.a: extract the domain and slots to query the DB
        sys_act = db_search(dialogue)

        # Task 2.b: generate system act (context + pred_belief + system actions)
        dialogue += ' <|bosys_act|> ' + sys_act + ' <|eosys_act|>'

        # print("system act================================")
        # print(dialogue)
        # print("system act================================")

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
            dec_dialogue.append(dialogue)
            print("After decoding ================")
            print(dialogue)

    return dec_dialogue

def db_search(context_pred_belief):

    loc_db_req = context_pred_belief.find('<|DB_req|>')
    loc_db_opt = context_pred_belief.find('<|DB_opt|>')
    loc_t_req = context_pred_belief.find('<|T_req|>')
    loc_t_opt = context_pred_belief.find('<|T_opt|>')
    loc_eob = context_pred_belief.find('<|eob|>')
    # get db_req slots
    if loc_db_req > 0:
        if loc_db_opt > 0:
            db_req = context_pred_belief[loc_db_req + 10:loc_db_opt]
        elif loc_t_req > 0:
            db_req = context_pred_belief[loc_db_req + 10:loc_t_req]
        elif loc_t_opt > 0:
            db_req = context_pred_belief[loc_db_req + 10:loc_t_opt]
        else:
            db_req = context_pred_belief[loc_db_req + 10:loc_eob]

        db_req = [x for x in list(dict.fromkeys(db_req.split(' '))) if x]

    # # get db_opt slots
    # if loc_db_opt > 0:
    #     if loc_t_req > 0:
    #         db_opt = context_pred_belief[loc_db_opt + 10:loc_t_req]
    #     elif loc_t_opt > 0:
    #         db_opt = context_pred_belief[loc_db_opt + 10:loc_t_opt]
    #     else:
    #         db_opt = context_pred_belief[loc_db_opt + 10:loc_eob]
    #     db_opt = list(dict.fromkeys(db_opt.split(' '))).remove('')

    # get t_req slots
    if loc_t_req > 0:
        if loc_t_opt > 0:
            t_req = context_pred_belief[loc_t_req + 9:loc_t_opt]
        else:
            t_req = context_pred_belief[loc_t_req + 9:loc_eob]

        t_req = [x for x in list(dict.fromkeys(t_req.split(' '))) if x]

    # # get t_opt slots
    # if loc_t_opt > 0:
    #     if loc_eob > 0:
    #         t_opt = context_pred_belief[loc_t_opt + 9:loc_eob]
    #         t_opt = list(dict.fromkeys(t_opt.split(' '))).remove('')

    results = ''
    # db_req slots
    if loc_db_req > 0:
        # get the domain
        domain = db_req[0]
        if domain == 'delivery':
            # set the condition for searching area_location table
            area, location = '', ''
            for item in db_req[1:]:
                key, value = item[0:item.find('=')], item[item.find('=') + 1:]
                if key == 'area':
                    area = value
                elif key == 'location':
                    location = value
            if (area == 'not_mentioned') and (location == 'not_mentioned'):
                results = 'area=null location=null'
            elif (area != 'not_mentioned') and (location == 'not_mentioned'):
                db_results = query_area(cfg.dataset_path_production_db, area)
                results = 'area=' + db_results + ' location=null'
            elif (area != 'not_mentioned') and (location != 'not_mentioned'):
                db_results = query_area_location(cfg.dataset_path_production_db, area, location)
                results = 'area=' + db_results + ' location=' + db_results
            elif (area == 'not_mentioned') and (location != 'not_mentioned'):
                db_results = query_location(cfg.dataset_path_production_db, location)
                results = 'area=null' + ' location=' + db_results

        elif domain == 'assembly':
            # set the condition for searching product table
            producttype = ''
            for item in db_req[1:]:
                key, value = item[0:item.find('=')], item[item.find('=') + 1:]
                if key == 'producttype':
                    producttype = value
            if (producttype == 'not_mentioned'):
                results = 'producttype=null'
            else:
                db_results, _ = query_product(cfg.dataset_path_production_db, producttype)
                results = 'producttype=' + db_results

    # t_req slots
    if loc_t_req > 0:
        # get the domain
        domain = t_req[0]
        if domain == 'delivery':
            # set the condition for searching area_location table
            object = ''
            for item in t_req[1:]:
                key, value = item[0:item.find('=')], item[item.find('=') + 1:]
                if key == 'object':
                    object = value
            if (object == 'not_mentioned'):
                results += ' object=null'
            else:
                results += ' object=' + object

        elif domain == 'assembly':
            # set the condition for searching product table
            product, quantity = '', ''
            for item in t_req[1:]:
                key, value = item[0:item.find('=')], item[item.find('=') + 1:]
                if key == 'product':
                    product = value
                elif key == 'quantity':
                    quantity = value
            if (product == 'not_mentioned'):
                results += ' product=null'
            else:
                results += ' product=' + product
            if (quantity == 'not_mentioned'):
                results += ' quantity=null'
            else:
                results += ' quantity=' + quantity

    return results


if __name__ == '__main__':

    f = cfg.dataset_path_test_file

    with open(f, 'r', encoding='utf-8') as test_file:
        lines = [line for line in test_file.read().splitlines() if (len(line) > 0 and not line.isspace())]

    results = decoder(lines)
    decoded_file = open(cfg.dataset_path_decoded_file, mode='wt', encoding='utf-8')
    for row in results:
        decoded_file.write(row + "\n")
    decoded_file.close()