from config import Config
# from utils.speech_service import *
from utils.dbsearch import *
from utils.normalize_text import normalize

import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM

cfg = Config()

def main():
    # load tokenizer from pretrained model
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.model_gpt_neo_checkpoint_path, bos_token='<|endoftext|>',
                                              eos_token='<|endoftext|>', pad_token='<|pad|>')
    # load fine-tuned GPT-neo model
    model = GPTNeoForCausalLM.from_pretrained(cfg.model_gpt_neo_checkpoint_path)
    # since the model is too large, I have to use cpu on my computer
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model.to(device)

    model.eval()

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
            pre_dialogue = dialogue
            if loc_eoc < 0:
                # the beginning of the dialogue
                #save the previous dialogue just in case it is not correctly understanded
                dialogue += '<|user|> ' + text + " <|eoc|>"
            else:
                # during the conversation
                tmp_context = dialogue.split('<|eoc|>', 1)
                dialogue = tmp_context[0] + ' <|user|> ' + text + tmp_context[1]

            # Task 1: generate belief
            with torch.no_grad():
                # encoding the dialogue
                dlg_ctx = tokenizer(dialogue,return_tensors="pt").input_ids.to(device)
                # get the highest probability response
                gen_res = model.generate(dlg_ctx, do_sample=True, top_k=50,
                                                max_length=1024, top_p=0.95, temperature=1.9, num_return_sequences=1)
                # generate the response
                gen_res = tokenizer.decode(gen_res[0], skip_special_tokens=True)
                # generate the belief
                temp_eob = gen_res.find('<|eob|>')
                if temp_eob > 0:
                    dialogue = gen_res[:temp_eob + 7]
                else:
                    response = "Sorry, I did not catch that. Can you repeat it?"
                    print("Max: " + response)
                    response = ""
                    # back to the previous dialogue context
                    dialogue = pre_dialogue
                    continue

            # Task 2.a: extract the domain and slots to query the DB
            sys_act = db_search(cfg, dialogue)

            # Task 2.b: generate context + pred_belief + system actions
            dialogue += ' <|bosys_act|> ' + sys_act + ' <|eosys_act|>'

            # Task 3: generation response (system + small talk)
            with torch.no_grad():
                # encoding the dialogue
                dlg_ctx = tokenizer(dialogue,return_tensors="pt").input_ids.to(device)
                # get the highest probability response
                gen_res = model.generate(dlg_ctx, do_sample=True, top_k=50,
                                                max_length=1024, top_p=0.95, temperature=1.9, num_return_sequences=1)
                # generate the response
                dialogue = tokenizer.decode(gen_res[0], skip_special_tokens=True)

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
    main()