import random

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
import argparse
from config import Config
from utils.speech_service import *
from utils.dbsearch import *
from utils.normalize_text import normalize

cfg = Config()

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


# def main():
#     parser = argparse.ArgumentParser()
#
#     ## Required parameters
#     parser.add_argument("--eval_data_file", default=None, type=str, required=True,
#                         help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
#
#     parser.add_argument("--output_dir", default=None, type=str, required=True,
#                         help="The output directory where the model predictions and checkpoints will be written.")
#
#     parser.add_argument("--model_type", default="gpt2", type=str,
#                         help="The model architecture to be fine-tuned.")
#
#     parser.add_argument("--model_name_or_path", default="", type=str,
#                         help="The model checkpoint for weights initialization. ToB4IR is trained from scratch. It does not need this.")
#
#     parser.add_argument('--decoding', default='greedy',
#                         help='decoding method for now.')
#
#     parser.add_argument("--config_name", default="", type=str,
#                         help="Optional pretrained config name or path if not the same as model_name_or_path")
#
#     parser.add_argument("--cache_dir", default=None, type=str,
#                         help="Optional directory to store the downloaded pre-trained models")
#
#     parser.add_argument("--block_size", default=1024, type=int,
#                         help="The default length of input sequnce is the max input 1024.")
#
#     parser.add_argument("--max_steps", default=-1, type=int,
#                         help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
#
#     parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
#
#     parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
#
#     parser.add_argument("--eval_all_checkpoints", action='store_true',
#                         help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
#
#     parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
#                         help="Batch size per GPU/CPU for training.")
#
#     parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
#                         help="Batch size per GPU/CPU for evaluation.")
#
#     parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
#                         help="Number of updates steps to accumulate before performing a backward/update pass.")
#
#     parser.add_argument("--learning_rate", default=5e-5, type=float,
#                         help="The initial learning rate for Adam.")
#
#     parser.add_argument("--weight_decay", default=0.0, type=float,
#                         help="Weight deay if we apply some.")
#
#     parser.add_argument("--adam_epsilon", default=1e-8, type=float,
#                         help="Epsilon for Adam optimizer.")
#
#     parser.add_argument("--max_grad_norm", default=1.0, type=float,
#                         help="Max gradient norm.")
#
#     parser.add_argument("--warmup_steps", default=0, type=int,
#                         help="Linear warmup over warmup_steps.")
#
#     parser.add_argument("--num_train_epochs", default=10, type=float,
#                         help="Total number of training epochs to perform.")
#
#     parser.add_argument('--logging_steps', type=int, default=100,
#                         help="Log every 100 updates steps.")
#
#     parser.add_argument('--save_steps', type=int, default=5000,
#                         help="Save checkpoint every 5000 updates steps.")
#
#     parser.add_argument("--no_cuda", action='store_true',
#                         help="Avoid using CUDA when available")
#
#     parser.add_argument('--seed', type=int, default=10,
#                         help="random seed for initialization")
#
#     parser.add_argument('--fp16', action='store_true',
#                         help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
#
#     parser.add_argument("--local_rank", type=int, default=-1,
#                         help="For distributed training: local_rank")
#
#     parser.add_argument('--save_total_limit', type=int, default=None,
#                         help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
#
#     args = parser.parse_args()


def do_generation():
    # load model
    model = GPT2LMHeadModel.from_pretrained(cfg.model_checkpoint_path)
    # get tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_checkpoint_path)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    endoftext = tokenizer.encode(tokenizer._eos_token)
    dialogue = '<|endoftext|> <|boc|> '
    while True:
        # get user input
        text_to_speech_microsoft("speak something")
        text = speech_to_text_microsoft().strip()
        # text = 'I want you to help me to deliver it to the lab. <|eoc|>'
        if len(text) > 0:
            text = normalize('i have a package here . i want you to help me to deliver it to the lab .')
            # text = normalize(text)
            # end of the current dialogue and go to next one
            if any(key in text for key in cfg.stop_words):
                dialogue = '<|endoftext|> <|boc|> '
                text_to_speech_microsoft(random.choice(cfg.max_end_dialogue))
                continue

            # add the user utterance to the previous context the "<|eoc|>" location
            loc_eoc = dialogue.find("<|eoc|>")
            if loc_eoc < 0:
                # the beginning of the dialogue
                dialogue += ' <|user|> ' + text + " <|eoc|>"
            else:
                # during the conversation
                tmp_context = dialogue.split('<|eoc|>', 1)
                dialogue = tmp_context[0] + ' <|user|> ' + text + tmp_context[1]

            # encoding the dialogue
            # # tempt = '<|endoftext|> <|boc|> <|user|> I want you to help me to deliver it to the lab . <|eoc|>'
            dialogue = '<|endoftext|> <|boc|> <|user|> i have a package here . i want you to help me to deliver it to the lab . <|eoc|>'
            indexed_tokens = tokenizer.encode(dialogue)
            if len(indexed_tokens) > cfg.max_length:
                indexed_tokens = indexed_tokens[-1 * cfg.max_length:]
            tokens_tensor = torch.tensor([indexed_tokens])
            tokens_tensor = tokens_tensor.to(device)
            predicted_index = indexed_tokens[-1]

            # generate belief
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
            print(dialogue)
            # text_to_speech_microsoft(context_pred_belief)

            # extract the domain and slots to query the DB and generate system act
            sys_act = db_search(dialogue)

            # context + pred_belief + system actions
            dialogue += ' <|bosys_act|> ' + sys_act + ' |eosys_act|'

            # test
            print(dialogue)

            # generation response
            indexed_tokens = tokenizer.encode(dialogue)
            if len(indexed_tokens) > cfg.max_length:
                indexed_tokens = indexed_tokens[-1 * cfg.max_length:]

            # Convert indexed tokens in a PyTorch tensor
            tokens_tensor = torch.tensor([indexed_tokens])

            tokens_tensor = tokens_tensor.to(device)
            predicted_index = indexed_tokens[-1]

            # Predict response
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

                # print the intermediate results
                print(dialogue)

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

                dialogue = new_dialogue + ' <|eoc|>'
                text_to_speech_microsoft(response)


if __name__ == "__main__":
    do_generation()
