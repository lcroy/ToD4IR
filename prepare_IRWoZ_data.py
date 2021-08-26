import json
from tqdm import tqdm

from config import Config
from utils.normalize_text import normalize

from transformers import GPT2Tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def lex_IRWoZ_data(raw_data, lex_data):
    # fin1 = file('data/multi-woz/data.json')
    #
    # data = json.load(fin1)
    # read the raw IRWoZ data
    IRWoZ_data = json.load(open(raw_data, 'r'))

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


            #write belief
            belief = ' <|bob|> '
            for i, domain in enumerate(turn['slots']):
                # find the domain
                if domain == dialogue_domain:
                    # write the db search required slots
                    temp_db_req = ""
                    db_req = turn['slots'][domain]['DB_request']['req']
                    for key, value in db_req.items():
                        temp_db_req += key + "=" + value + " "
                    belief += ' <|DB_req|> ' + dialogue_domain + " " + temp_db_req
                    # write the db search optional slots
                    temp_db_opt = ""
                    db_opt = turn['slots'][domain]['DB_request']['opt']
                    for key, value in db_opt.items():
                        if value != "not mentioned":
                            temp_db_opt += key + "=" + value + " "
                    belief += ' <|DB_opt|> ' + dialogue_domain + " " + temp_db_opt

                    # write the task related info slots
                    temp_t_req = ""
                    t_req = turn['slots'][domain]['T_inform']['req']
                    for key, value in t_req.items():
                        temp_t_req += key + "=" + value + " "
                    belief += ' <|T_req|> ' + dialogue_domain + " " + temp_t_req
                    # write the db search optional slots
                    temp_t_opt = ""
                    t_opt = turn['slots'][domain]['T_inform']['opt']
                    for key, value in t_opt.items():
                        if value != "not mentioned":
                            temp_t_opt += key + "=" + value + " "
                    belief += ' <|T_opt|> ' + dialogue_domain + " " + temp_t_opt
            belief += ' <|eob|> '

            #write system act
            sys_act = ' <|sys_act|> '
            for key, value in turn['search_result'].items():
                sys_act += key + "=" + value + " "

            # write sys + small talk response
            temp_t_res = normalize(turn['system'])
            temp_r_res = normalize(turn['s_system'])
            t_res = ' <|boTres|> ' + temp_t_res + ' <|eoTres|>'
            s_res = ' <|boSres|> ' + temp_r_res + ' <|eoSres|>'

            # final text
            text = context + belief + sys_act + t_res + s_res

            with open(lex_data, 'at', encoding='utf-8') as f:
                f.write('{} {} {}\n'.format(gpt2_tokenizer._bos_token, text, gpt2_tokenizer._bos_token))


def delex_IRWoZ_data(dataset_path):
    # read the raw IRWoZ data
    IRWoZ_data = json.load(open(dataset_path, 'r'))

    items = IRWoZ_data.items()
    for key, value in items:
        print(str(key) + '=' + str(value))


def main():

    cfg = Config()

    # format the raw dialogues to lex data
    lex_data = lex_IRWoZ_data(cfg.dataset_path_IR, cfg.dataset_path_IR_lex)

    # format the raw dialogues to lex data
    # dlex_data = delex_IRWoZ_data(lex_data)


if __name__ == "__main__":
    main()


#
#
# from utils.args_parser import ArgsParser
# from data.dataset.multiwoz import MultiWozDataset
# import en_core_web_sm
# from nltk import ngrams
# from utils.multiwoz import dbPointer
# import ipdb
# import json
# import random
# import os
#
# from transformers import GPT2Tokenizer
# gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#
# multiwoz_data = json.load(open('resources/multi-woz/lex.json', 'r'))
# save_dir = './resources/gpt2'
# os.makedirs(save_dir, exist_ok=True)
#
# for split in ['train', 'val', 'test']:
#
#     opt = ArgsParser().parse()
#     opt.use_knowledge = True
#     opt.use_action = True
#     opt.context_knowledge = True
#     opt.lexical = True
#
#     data = MultiWozDataset(opt, split=split, shuffle=False)
#
#     opt_delex = ArgsParser().parse()
#     data_delex = MultiWozDataset(opt_delex, split=split, shuffle=False)
#
#     history_raw_new = []
#     belief_raw_new = []
#     belief_raw_none_new = []
#     action_raw_new = []
#     output_raw_new = []
#     output_raw_delex_new = []
#     db_search_raw = []
#     db_nmatch_raw = []
#
#     if split == 'test':
#         test_dict = {}
#
#     lex_dict = {}
#     delex_dict = {}
#     for d in data:
#         lex_dict[d['name']] = d
#
#     for d in data_delex:
#         delex_dict[d['name']] = d
#
#     for key in lex_dict:
#         d_lex = lex_dict[key]
#         d_delex = delex_dict[key]
#         inp = d_lex['input_raw']
#         out = d_lex['target_raw']
#         out_delex = d_delex['target_raw']
#         db_data = d_lex['db']
#         goal = multiwoz_data[key]['goal']
#
#         for i, (usr, sys) in enumerate(zip(inp, out)):
#             if i == 0:
#                 history_new = '<|context|> <|user|> {} <|endofcontext|>'.format(usr)
#             else:
#                 tmp_new = ['<|context|>']
#                 for k in range(i):
#
#                     tmp_new.append('<|user|> ' + inp[k])
#                     tmp_new.append('<|system|> ' + out[k])
#
#                 tmp_new.append('<|user|> ' + usr + '<|endofcontext|>')
#                 history_new = ' '.join(tmp_new)
#
#             sys_delex = out_delex[i]
#             history_raw_new.append(history_new)
#             output_raw_new.append('<|response|> ' + sys + ' <|endofresponse|>')
#
#             output_raw_delex_new.append('<|response|> ' + sys_delex.strip() + ' <|endofresponse|>')
#
#             db_text = dbPointer.convert_dbpointer_to_text(db_data[i], goal, d_lex['belief_raw'][i])
#             db_search_raw.append('<|dbsearch|> {} <|endofdbsearch|>'.format(db_text))
#
#             db_text_nmatch = dbPointer.convert_dbpointer_to_text_nmatch(db_data[i], goal, d_lex['belief_raw'][i])
#             db_nmatch_raw.append('<|dbsearch|> {} <|endofdbsearch|>'.format(db_text_nmatch))
#
#         belief = d_lex['belief_raw']
#         for bs in belief:
#             tmp_bs_new = []
#             for i, b in enumerate(bs):
#                 if b[-1] in ['not mentioned']: # comment this for DST task
#                     continue
#                 if i == len(bs) - 1:
#                     tmp_bs_new.append(' '.join(b))
#                 else:
#                     tmp_bs_new.append(' '.join(b))
#
#             if len(tmp_bs_new) == 0:
#                 tmp_bs_new.append(' ')
#
#             tmp_new = '<|belief|> {} <|endofbelief|>'.format(' , '.join(tmp_bs_new))
#             belief_raw_new.append(tmp_new)
#
#         # belief for DST task (include none)
#         for bs in belief:
#             tmp_bs_new = []
#             for i, b in enumerate(bs):
#                 if i == len(bs) - 1:
#                     tmp_bs_new.append(' '.join(b))
#                 else:
#                     tmp_bs_new.append(' '.join(b))
#
#             if len(tmp_bs_new) == 0:
#                 tmp_bs_new.append(' ')
#
#             tmp_new = '<|belief|> {} <|endofbelief|>'.format(' , '.join(tmp_bs_new))
#             belief_raw_none_new.append(tmp_new)
#
#         action = d_lex['action_raw']
#         for act in action:
#             tmp_act_new = []
#             for i, a in enumerate(act):
#                 if i == len(act) - 1:
#                     tmp_act_new.append(' '.join(a))
#                 else:
#                     tmp_act_new.append(' '.join(a))
#             if len(tmp_act_new) == 0:
#                 tmp_act_new.append(' ')
#
#             tmp_new = '<|action|> {} <|endofaction|>'.format(' , '.join(tmp_act_new))
#             action_raw_new.append(tmp_new)
#
#     tmp = []
#     for inp, bs, dbsearch, act, trg in zip(history_raw_new, belief_raw_new, db_search_raw, action_raw_new, output_raw_delex_new):
#         tmp.append(' '.join([inp.lower(), bs.lower(), dbsearch.lower(), act, trg]))
#     with open('{}/{}.history_belief_dbsearch_action_sys_delex'.format(save_dir, split), 'wt', encoding='utf-8') as f:
#         for l in tmp:
#             f.write('{} {}\n'.format(gpt2_tokenizer._bos_token, l.lower()))
#
#     tmp = []
#     for inp, bs, dbsearch, act, trg in zip(history_raw_new, belief_raw_new, db_nmatch_raw, action_raw_new,
#                                            output_raw_delex_new):
#         tmp.append(' '.join([inp.lower(), bs.lower(), dbsearch.lower(), act, trg]))
#     with open('{}/{}.history_belief_dbnmatch_action_sys_delex'.format(save_dir, split), 'wt', encoding='utf-8') as f:
#         for l in tmp:
#             f.write('{} {}\n'.format(gpt2_tokenizer._bos_token, l.lower()))
#
#     with open('{}/{}.history'.format(save_dir, split), 'wt', encoding='utf-8') as f:
#         for l in history_raw_new:
#             f.write('{} {}\n'.format(gpt2_tokenizer._bos_token, l.lower()))
#
#     tmp = []
#     for hist, bs in zip(history_raw_new, belief_raw_none_new):
#         tmp.append(' '.join([hist.lower(), bs.lower()]))
#     with open('{}/{}.history_belief'.format(save_dir, split),
#               'wt', encoding='utf-8') as f:
#         for l in tmp:
#             f.write('{} {} {}\n'.format(gpt2_tokenizer._bos_token, l.lower(), gpt2_tokenizer._eos_token))
#
#     tmp = []
#     for hist, bs, act, trg in zip(history_raw_new, belief_raw_new, action_raw_new, output_raw_delex_new):
#         tmp.append(' '.join([hist.lower(), bs.lower(), act, trg]))
#     with open('{}/{}.history_belief_action_sys_delex'.format(save_dir, split), 'wt', encoding='utf-8') as f:
#         for l in tmp:
#             f.write('{} {} {}\n'.format(gpt2_tokenizer._bos_token, l.lower(), gpt2_tokenizer._eos_token))
#
