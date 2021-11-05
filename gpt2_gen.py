import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from config import Config
# from utils.speech_service import *
from utils.dbsearch import *
from utils.normalize_text import normalize

cfg = Config()


def do_generation():
    # load model
    model = GPT2LMHeadModel.from_pretrained(cfg.model_gpt2_checkpoint_path)
    # get tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_gpt2_checkpoint_path)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    endoftext = tokenizer.encode(tokenizer._eos_token)
    dialogue = '<|endoftext|> <|boc|> '
    while True:
        # get user input
        # text_to_speech_microsoft("speak something")
        # text = speech_to_text_microsoft().strip()
        # text = 'I want you to help me to deliver it to the lab. <|eoc|>'
        text = input("User: ")
        if len(text) > 0:
            # text = normalize('i have a package here . i want you to help me to deliver it to the lab .')
            text = normalize(text)
            # end of the current dialogue and go to next one
            if any(key in text for key in cfg.stop_words):
                dialogue = '<|endoftext|> <|boc|> '
                # text_to_speech_microsoft(random.choice(cfg.max_end_dialogue))
                continue

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
            # # tempt = '<|endoftext|> <|boc|> <|user|> I want you to help me to deliver it to the lab . <|eoc|>'
            # dialogue = '<|endoftext|> <|boc|> <|user|> i have a package here . i want you to help me to deliver it to the lab . <|eoc|>'
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
            print("belief================================")
            print(dialogue)
            print("belief================================")

            # Task 2.a: extract the domain and slots to query the DB
            sys_act = db_search(cfg, dialogue)

            # Task 2.b: generate system act (context + pred_belief + system actions)
            dialogue += ' <|bosys_act|> ' + sys_act + ' <|eosys_act|>'

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

                dialogue = new_dialogue + ' <|eoc|>'
                # text_to_speech_microsoft(response)
                print("Max: " + response)


if __name__ == "__main__":
    do_generation()
