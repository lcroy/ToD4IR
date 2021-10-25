import math
from collections import Counter
from nltk.util import ngrams
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from config import Config
from utils.normalize_text import normalize

cfg = Config()

class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def score(self, hypothesis, corpus, n=1):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [[1, 0, 0, 0],[0.5, 0.5, 0, 0],[0.33, 0.33, 0.33, 0],[0.25, 0.25, 0.25, 0.25]]

        # accumulate ngram statistics
        for hyps, refs in zip(hypothesis, corpus):

            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]

            for idx, hyp in enumerate(hyps):
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)
                if n == 1:
                    break
        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        bleu = []
        for i in range(4):
            s = math.fsum(w * math.log(p_n) \
                          for w, p_n in zip(weights[i], p_ns) if p_n)
            bleu.append(bp * math.exp(s))

        return bleu


# load the decoded file (i.e., generated dialogue corpus) and test file (i.e., reference dialogue corpus)
def load_dialogue_corpus():
    # load test file
    try:
        with open(cfg.dataset_path_test_file, encoding="utf-8") as f:
            ref_dialogue = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    except:
        print("The test file does not exists. Please make sure you have it on your driver.")

    # load decoded file
    try:
        with open(cfg.dataset_path_decoded_file, encoding="utf-8") as f:
            gen_dialogue = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    except:
        print("The decoded file does not exists. Please make sure you have it on your driver.")

    return gen_dialogue, ref_dialogue


# Evaluate DST: Joint goal accuracy and slot accuracy
def evaluate_DST(gen_dialogue, ref_dialogue):
    num_gen = len(gen_dialogue)
    num_ref = len(ref_dialogue)

    num_matched_joint_ga = 0
    joint_ga = 0.0

    num_matched_slot = 0
    num_mismatched_slot = 0
    slot_acc = 0.0

    if num_gen != num_ref:
        print("The number of decoded dialogue turn is {0} and test dialogue trun is {1}".format(num_gen,num_ref))
        print("The number does not match.")
    else:
        for i in range(num_gen):
            # joint goal accuracy compares the predicted dialog states to the ground truth at each dialog turn,
            # and the output is considered correct if and only if all the predicted values exactly match the ground truth.
            gen_loc_bob = gen_dialogue[i].find('<|bob|>')
            gen_loc_eob = gen_dialogue[i].find('<|eob|>')
            gen_belief = gen_dialogue[i][gen_loc_bob + 8: gen_loc_eob - 1 ]

            ref_loc_bob = ref_dialogue[i].find('<|bob|>')
            ref_loc_eob = ref_dialogue[i].find('<|eob|>')
            ref_belief = ref_dialogue[i][ref_loc_bob + 8: ref_loc_eob - 1]

            if gen_belief == ref_belief:
                num_matched_joint_ga += 1


            # slot accuracy individually compares each (domain, slot, value) triplet to its ground truth label
            # we compare each DB_req, DB_opt, T_req, T_opt to its ground truth label
            # decoded file
            gen_loc_db_req = gen_dialogue[i].find('<|DB_req|>')
            gen_loc_db_opt = gen_dialogue[i].find('<|DB_opt|>')
            gen_loc_t_req = gen_dialogue[i].find('<|T_req|>')
            gen_loc_t_opt = gen_dialogue[i].find('<|T_opt|>')
            gen_loc_eob = gen_dialogue[i].find('<|eob|>')

            # if DB_opt is generated
            if gen_loc_db_opt > 0:
                gen_db_req = gen_dialogue[i][gen_loc_db_req + 11: gen_loc_db_opt - 1 ]
                gen_db_opt = gen_dialogue[i][gen_loc_db_opt + 11: gen_loc_t_req - 1]
            else:
                gen_db_req = gen_dialogue[i][gen_loc_db_req + 11: gen_loc_t_req - 1]
                gen_db_opt = "None"

            # if T_opt is generated
            if gen_loc_t_opt > 0:
                gen_t_req = gen_dialogue[i][gen_loc_t_req + 11: gen_loc_t_opt - 1]
                gen_t_opt = gen_dialogue[i][gen_loc_t_opt + 11: gen_loc_eob - 1]
            else:
                gen_t_req = gen_dialogue[i][gen_loc_t_req + 11: gen_loc_eob - 1]
                gen_t_opt = "None"

            # test file
            ref_loc_db_req = ref_dialogue[i].find('<|DB_req|>')
            ref_loc_db_opt = ref_dialogue[i].find('<|DB_opt|>')
            ref_loc_t_req = ref_dialogue[i].find('<|T_req|>')
            ref_loc_t_opt = ref_dialogue[i].find('<|T_opt|>')
            ref_loc_eob = ref_dialogue[i].find('<|eob|>')

            # if DB_opt is generated
            if ref_loc_db_opt > 0:
                ref_db_req = ref_dialogue[i][ref_loc_db_req + 11: ref_loc_db_opt - 1]
                ref_db_opt = ref_dialogue[i][ref_loc_db_opt + 11: ref_loc_t_req - 1]
            else:
                ref_db_req = ref_dialogue[i][ref_loc_db_req + 11: ref_loc_t_req - 1]
                ref_db_opt = "None"

            # if T_opt is generated
            if ref_loc_t_opt > 0:
                ref_t_req = ref_dialogue[i][ref_loc_t_req + 11: ref_loc_t_opt - 1]
                ref_t_opt = ref_dialogue[i][ref_loc_t_opt + 11: ref_loc_eob - 1]
            else:
                ref_t_req = ref_dialogue[i][ref_loc_t_req + 11: ref_loc_eob - 1]
                ref_t_opt = "None"

            if gen_db_req == ref_db_req:
                num_matched_slot += 1
            else:
                num_mismatched_slot += 1

            if gen_db_opt == ref_db_opt:
                num_matched_slot += 1
            else:
                num_mismatched_slot += 1

            if gen_t_req == ref_t_req:
                num_matched_slot += 1
            else:
                num_mismatched_slot += 1

            if gen_t_opt == ref_t_opt:
                num_matched_slot += 1
            else:
                num_mismatched_slot += 1


        # joint goal accuracy and slot accuracy
        if num_gen > 0:
            joint_ga = float(num_matched_joint_ga)/float(num_gen)
            slot_acc = float(num_matched_slot)/float(num_matched_slot + num_mismatched_slot)
            print("The joint goal accuracy is {0:.3f}".format(joint_ga))
            print("The slot accuracy is {0:.3f}".format(slot_acc))
        else:
            print("The decoded file is empty.")



# Evaluate Dialogue context-to-text generation: BLEU and slot error rate
def evaluate_generated_dialogue(gen_dialogue, ref_dialogue):

    corpus = []
    model_corpus = []
    bscorer = BLEUScorer()

    model_turns, corpus_turns = [], []
    for i in range(len(gen_dialogue)):
        gen_loc_boTres = gen_dialogue[i].find('<|boTres|>')
        gen_loc_eoTres = gen_dialogue[i].find('<|eoTres|>')
        gen_loc_boSres = gen_dialogue[i].find('<|boSres|>')
        gen_loc_eoSres = gen_dialogue[i].find('<|eoSres|>')
        gen_loc_endoftext = gen_dialogue[i].find('<|endoftext|>')

        gen_res = gen_dialogue[i][gen_loc_boTres + 11:gen_loc_eoTres-1] + " " + gen_dialogue[i][gen_loc_boSres + 11:gen_loc_eoSres-1]
        gen_res = normalize(gen_res.replace("<|endoftext",""))
        model_turns.append([gen_res])

        # lexical test file response (replace all [slot] to belief state)
        while ref_dialogue[i].find('[') > 0:
            # get location of delex
            delex_slot = ref_dialogue[i][ref_dialogue[i].find('[')+1:ref_dialogue[i].find(']')]
            # get slot value
            temp_ref = ref_dialogue[i][ref_dialogue[i].find(delex_slot)+len(delex_slot)+1:]
            slot_val = temp_ref[:temp_ref.find(' ')]
            # replace delex position
            temp = ref_dialogue[i][:ref_dialogue[i].find('[')] + slot_val + ref_dialogue[i][ref_dialogue[i].find(']') + 1:]
            ref_dialogue[i] = temp

        ref_loc_boTres = ref_dialogue[i].find('<|boTres|>')
        ref_loc_eoTres = ref_dialogue[i].find('<|eoTres|>')
        ref_loc_boSres = ref_dialogue[i].find('<|boSres|>')
        ref_loc_eoSres = ref_dialogue[i].find('<|eoSres|>')
        ref_loc_endoftext = ref_dialogue[i].find('<|endoftext|>')

        ref_res = ref_dialogue[i][ref_loc_boTres + 11:ref_loc_eoTres - 1] + " " + ref_dialogue[i][
                                                                            ref_loc_boSres + 11:ref_loc_eoSres - 1]
        ref_res = normalize(ref_res)
        corpus_turns.append([ref_res])

    corpus.extend(corpus_turns)
    model_corpus.extend(model_turns)

    blue_score = bscorer.score(model_corpus, corpus)

    for i in range(4):
        print("The BLEU-{} is : {:.4f}".format(i+1,blue_score[i]))


if __name__ == '__main__':
    # load dialogue data from decoded and test files
    gen_dialogue, ref_dialogue = load_dialogue_corpus()

    # evaluate DST
    evaluate_DST(gen_dialogue, ref_dialogue)

    # evaluate Dialogue context-to-text generation
    evaluate_generated_dialogue(gen_dialogue, ref_dialogue)
