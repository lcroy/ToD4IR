import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
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
        sys_act = db_search(cfg, dialogue)

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


if __name__ == '__main__':

    f = cfg.dataset_path_test_file

    with open(f, 'r', encoding='utf-8') as test_file:
        lines = [line for line in test_file.read().splitlines() if (len(line) > 0 and not line.isspace())]

    results = decoder(lines)
    decoded_file = open(cfg.dataset_path_decoded_file, mode='wt', encoding='utf-8')
    for row in results:
        decoded_file.write(row + "\n")
    decoded_file.close()