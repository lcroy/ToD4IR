import json
from tqdm import tqdm
from config import Config
from utils.normalize_text import normalize

from transformers import GPT2Tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def delex_IRWoZ_data(raw_data, delex_data):

    IRWoZ_data = json.load(open(raw_data, 'rb'))

    # samples, context, belief, search result
    context = ""
    temp_t_res = ""
    temp_s_res = ""

    # read the dialogue one by one
    for dialogue_name in tqdm(IRWoZ_data):
        dialogue = IRWoZ_data[dialogue_name]

        # get the dialogue domain
        key_list = list(dialogue['domain'].keys())
        val_list = list(dialogue['domain'].values())
        position = val_list.index(True)
        dialogue_domain = key_list[position]

        for idx, turn in enumerate(dialogue['turn']):
            # write context
            if idx == 0:
                user = '<|user|> ' + normalize(turn['user'])
                context = '<|boc|> ' + user + ' <|eoc|>'
            else:
                context = context.replace(' <|eoc|>', '')
                context += ' <|sys|> ' + pre_t_res + ' ' + pre_s_res + ' <|user|> ' + normalize(turn['user']) + ' <|eoc|>'

            # write sys + small talk response
            temp_t_res = normalize(turn['system'])
            temp_s_res = normalize(turn['s_system'])

            #write belief
            belief = ' <|bob|>'
            for i, domain in enumerate(turn['slots']):
                # find the domain
                if domain == dialogue_domain:

                    # write the db search required slots
                    temp_db_req = ""
                    db_req = turn['slots'][domain]['DB_request']['req']
                    for key, value in db_req.items():
                        temp_db_req += key + "=" + value + " "
                        # update t_res with delex representation
                        if ((value != "") & (value != "not_mentioned")):
                            temp_t_res = temp_t_res.replace(value, '[' + key + ']')
                    belief += ' <|DB_req|> ' + dialogue_domain + " " + temp_db_req

                    # write the db search optional slots
                    temp_db_opt = ""
                    opt_flag = 0
                    db_opt = turn['slots'][domain]['DB_request']['opt']
                    for key, value in db_opt.items():
                        if value != "not_mentioned":
                            temp_db_opt += key + "=" + value + " "
                            # update t_res with delex representation
                            if ((value != "") & (value != "not_mentioned")):
                                temp_t_res = temp_t_res.replace(value, '[' + key + ']')
                            opt_flag = 1
                    if opt_flag == 1:
                        belief += '<|DB_opt|> ' + dialogue_domain + " " + temp_db_opt

                    # write the task related info slots
                    temp_t_req = ""
                    t_req = turn['slots'][domain]['T_inform']['req']
                    for key, value in t_req.items():
                        temp_t_req += key + "=" + value + " "
                        # update t_res with delex representation
                        if ((value != "") & (value != "not_mentioned")):
                            temp_t_res = temp_t_res.replace(value, '[' + key + ']')
                    belief += '<|T_req|> ' + dialogue_domain + " " + temp_t_req

                    # write the db search optional slots
                    temp_t_opt = ""
                    opt_flag = 0
                    t_opt = turn['slots'][domain]['T_inform']['opt']
                    for key, value in t_opt.items():
                        if value != "not_mentioned":
                            temp_t_opt += key + "=" + value + " "
                            opt_flag = 1
                            # update t_res with delex representation
                            if ((value != "") & (value != "not_mentioned")):
                                temp_t_res = temp_t_res.replace(value, '[' + key + ']')
                    if opt_flag == 1:
                        belief += '<|T_opt|> ' + dialogue_domain + " " + temp_t_opt

            belief += '<|eob|> '

            #write system act
            sys_act = '<|bosys_act|> ' + dialogue_domain + ' '
            for key, value in turn['search_result'].items():
                sys_act += key + "=" + value + " "

            sys_act += '<|eosys_act|>'

            # save the current sys + small talk response for next round
            pre_t_res = normalize(turn['system'])
            pre_s_res = normalize(turn['s_system'])

            # create delex response
            temp_t_res = normalize(temp_t_res)
            temp_s_res = normalize(temp_s_res)
            t_res = ' <|boTres|> ' + temp_t_res + ' <|eoTres|>'
            s_res = ' <|boSres|> ' + temp_s_res + ' <|eoSres|>'

            # final text
            text = context + belief + sys_act + t_res + s_res

            with open(delex_data, 'at', encoding='utf-8') as f:
                f.write('{} {} {}\n'.format(gpt2_tokenizer._bos_token, text, gpt2_tokenizer._bos_token))


def pre_delex_IRWoZ_data(raw_data, pre_delex_data):

    IRWoZ_data = json.load(open(raw_data, 'rb'))

    # samples, context, belief, search result
    context = ""
    temp_t_res = ""
    temp_s_res = ""

    # read the dialogue one by one
    for dialogue_name in tqdm(IRWoZ_data):
        dialogue = IRWoZ_data[dialogue_name]

        # get the dialogue domain
        key_list = list(dialogue['domain'].keys())
        val_list = list(dialogue['domain'].values())
        position = val_list.index(True)
        dialogue_domain = key_list[position]

        for idx, turn in enumerate(dialogue['turn']):
            # write context
            if idx == 0:
                user = '<|user|> ' + normalize(turn['user'])
                context = '<|boc|> ' + user + ' <|eoc|>'
            else:
                context = context.replace(' <|eoc|>', '')
                context += ' <|sys|> ' + temp_t_res + temp_s_res + ' <|user|> ' + normalize(turn['user']) + ' <|eoc|>'

            # write sys + small talk response
            temp_t_res = normalize(turn['system'])
            temp_r_res = normalize(turn['s_system'])

            # write belief
            belief = ' <|bob|>'
            for i, domain in enumerate(turn['slots']):
                # find the domain
                if domain == dialogue_domain:
                    # write the db search required slots
                    temp_db_req = ""
                    db_req = turn['slots'][domain]['DB_request']['req']
                    for key, value in db_req.items():
                        temp_db_req += key + "=" + value + " "
                        #update t_res with delex representation
                        if ((value!="") & (value!="not_mentioned")):
                            temp_t_res = temp_t_res.replace(value, '[' + key + ']')
                    belief += ' <|DB_req|> ' + dialogue_domain + " " + temp_db_req

                    # write the db search optional slots
                    temp_db_opt = ""
                    db_opt = turn['slots'][domain]['DB_request']['opt']
                    opt_flg = 0
                    for key, value in db_opt.items():
                        if value != "not_mentioned":
                            temp_db_opt += key + "=" + value + " "
                            opt_flg = 1
                        #update t_res with delex representation
                        if ((value!="") & (value!="not_mentioned")):
                            temp_t_res = temp_t_res.replace(value, '[' + key + ']')
                    if opt_flg == 1:
                        belief += ' <|DB_opt|> ' + dialogue_domain + " " + temp_db_opt

                    # write the task related info slots
                    temp_t_req = ""
                    t_req = turn['slots'][domain]['T_inform']['req']
                    for key, value in t_req.items():
                        temp_t_req += key + "=" + value + " "
                        # update t_res with delex representation
                        if ((value != "") & (value != "not_mentioned")):
                            temp_t_res = temp_t_res.replace(value, '[' + key + ']')
                    belief += '<|T_req|> ' + dialogue_domain + " " + temp_t_req

                    # write the task related info slots
                    temp_t_opt = ""
                    t_opt = turn['slots'][domain]['T_inform']['opt']
                    opt_flg = 0
                    for key, value in t_opt.items():
                        if value != "not_mentioned":
                            temp_t_opt += key + "=" + value + " "
                            opt_flg = 1
                        # update t_res with delex representation
                        if ((value != "") & (value != "not_mentioned")):
                            temp_t_res = temp_t_res.replace(value, '[' + key + ']')

                    if opt_flg == 1:
                        belief += ' <|T_opt|> ' + dialogue_domain + " " + temp_t_opt

            belief += '<|eob|> '

            # updated system response
            t_res = ' <|boTres|> ' + temp_t_res + ' <|eoTres|>'
            s_res = ' <|boSres|> ' + temp_r_res + ' <|eoSres|>'

            # write system act
            sys_act = '<|bosys_act|> ' + dialogue_domain + ' '
            for key, value in turn['search_result'].items():
                sys_act += key + "=" + value + " "

            sys_act += '<|eosys_act|>'

            # final text
            text = context + belief + sys_act + t_res + s_res

            with open(pre_delex_data, 'at', encoding='utf-8') as f:
                f.write('{} {} {}\n'.format(gpt2_tokenizer._bos_token, text, gpt2_tokenizer._bos_token))


def main():
    cfg = Config()

    # format the raw dialogues to pre_delex data
    pre_delex_IRWoZ_data(cfg.dataset_path_IR, cfg.dataset_path_IR_pre_delex)

    # format the raw dialogues to delex data
    delex_IRWoZ_data(cfg.dataset_path_IR, cfg.dataset_path_IR_delex)


if __name__ == "__main__":
    main()
