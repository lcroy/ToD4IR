from config import Config
# from utils.speech_service import *
from utils.dbsearch import *
from utils.normalize_text import normalize

import torch
from transformers import GPT2Tokenizer, GPTNeoForCausalLM

cfg = Config()

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
#     # # get db_opt slots
#     # if loc_db_opt > 0:
#     #     if loc_t_req > 0:
#     #         db_opt = context_pred_belief[loc_db_opt + 10:loc_t_req]
#     #     elif loc_t_opt > 0:
#     #         db_opt = context_pred_belief[loc_db_opt + 10:loc_t_opt]
#     #     else:
#     #         db_opt = context_pred_belief[loc_db_opt + 10:loc_eob]
#     #     db_opt = list(dict.fromkeys(db_opt.split(' '))).remove('')
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
#     # # get t_opt slots
#     # if loc_t_opt > 0:
#     #     if loc_eob > 0:
#     #         t_opt = context_pred_belief[loc_t_opt + 9:loc_eob]
#     #         t_opt = list(dict.fromkeys(t_opt.split(' '))).remove('')
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
#                 results += ' object=' + object
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
#                 results += ' product=' + product
#             if (quantity == 'not_mentioned'):
#                 results += ' quantity=null'
#             else:
#                 results += ' quantity=' + quantity
#
#     return results

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

    # get t_req slots
    if loc_t_req > 0:
        if loc_t_opt > 0:
            t_req = context_pred_belief[loc_t_req + 9:loc_t_opt]
        else:
            t_req = context_pred_belief[loc_t_req + 9:loc_eob]

        t_req = [x for x in list(dict.fromkeys(t_req.split(' '))) if x]

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

        elif domain == 'position':
            # set the condition for searching position table
            position_name = ''
            for item in db_req[1:]:
                key, value = item[0:item.find('=')], item[item.find('=') + 1:]
                if key == 'position_name':
                    producttype = value
            if (position_name == 'not_mentioned'):
                results = 'position_name=null'
            else:
                db_results, _ = query_position_name(cfg.dataset_path_production_db, position_name)
                results = 'position_name=' + db_results

        elif domain == 'relocation':
            # set the condition for searching position table
            object_name = ''
            for item in db_req[1:]:
                key, value = item[0:item.find('=')], item[item.find('=') + 1:]
                if key == 'object_name':
                    object_name = value
            if (object_name == 'not_mentioned'):
                results = 'object_name=null'
            else:
                db_results, _ = query_object(cfg.dataset_path_production_db, object_name)
                results = 'object_name=' + db_results

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
                results += ' object=detected'

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
                results += ' product=detected'
            if (quantity == 'not_mentioned'):
                results += ' quantity=null'
            else:
                results += ' quantity=detected'

        elif domain == 'position':
            operation = ''
            for item in t_req[1:]:
                key, value = item[0:item.find('=')], item[item.find('=') + 1:]
                if key == 'operation':
                    operation = value
            if (operation == 'not_mentioned'):
                results += ' operation=null'
            else:
                results += ' operation=detected'

    return results

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
            sys_act = db_search(dialogue)

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