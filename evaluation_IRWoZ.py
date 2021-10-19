import math
import utils.delexicalize as delex
from collections import Counter
from nltk.util import ngrams
import json
from utils.nlp import normalize
import sqlite3
import os
import random
import logging
from utils.nlp import BLEUScorer
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu
from config import Config

cfg = Config()

class BaseEvaluator(object):
    def initialize(self):
        raise NotImplementedError

    def add_example(self, ref, hyp):
        raise NotImplementedError

    def get_report(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _get_prec_recall(tp, fp, fn):
        precision = tp / (tp + fp + 10e-20)
        recall = tp / (tp + fn + 10e-20)
        f1 = 2 * precision * recall / (precision + recall + 1e-20)
        return precision, recall, f1

    @staticmethod
    def _get_tp_fp_fn(label_list, pred_list):
        tp = len([t for t in pred_list if t in label_list])
        fp = max(0, len(pred_list) - tp)
        fn = max(0, len(label_list) - tp)
        return tp, fp, fn


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
        weights = [0.25, 0.25, 0.25, 0.25]

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
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu


class IRWOZ(object):
    # building database connection
    domains = ['Delivery', 'Position', 'Assembly', 'Relocation']
    db_conns = {}
    db = cfg.dataset_path_production_db
    for domain in domains:
        conn = sqlite3.connect(db)
        c = conn.cursor()
        db_conns[domain] = c

    def queryResultVenues(self, domain, turn, bs=None, real_belief=False):
        # query the db
        sql_query = "select * from {}".format(domain)

        if real_belief == True:
            items = turn.items()
        else:
            items = turn['turns'][slots]['domain']['DB_request'].items()

        # if bs is None:
        # return []

        if bs is not None:
            items = bs.items()
            #     # print(bs, turn.items())
            if len(items) == 0:
                return []
        flag = True
        for key, val in items:
            if val == "" or val == 'not_mentioned':
                pass
            else:
                if flag:
                    sql_query += " where "
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)

                    sql_query += r" " + key + "=" + r"'" + val2 + r"'"
                    flag = False
                else:
                    val2 = val.replace("'", "''")
                    val2 = normalize(val2)
                    sql_query += r" and " + key + "=" + r"'" + val2 + r"'"
        try:  # "select * from *  where * = '*'"
            return self.db_conns[domain].execute(sql_query).fetchall()
        except:
            return []  # TODO test it

class IRWOZEvaluator(BaseEvaluator):
    def __init__(self, mode, lines):
        self.mode = mode
        self.slot_dict = delex.prepareSlotValuesIndependent()
        self.text = lines
        self.db = IRWOZ()
        self.labels = list()
        self.hyps = list()

    def add_example(self, ref, hyp):
        self.labels.append(ref)
        self.hyps.append(hyp)

    def _evaluateGeneratedDialogue(self, dialog, goal, realDialogue, real_requestables, soft_acc=False):
        """Evaluates the dialogue created by the model.
        First we load the user goal of the dialogue, then for each turn
        generated by the system we look for key-words.
        For the Inform rate we look whether the entity was proposed.
        For the Success rate we look for requestables slots"""
        # for computing corpus success
        requestables = ['producttype', 'area', 'location']

        dialog, bs = dialog
        provided_requestables = {}
        venue_offered = {}
        domains_in_goal = []

        for domain in goal.keys():
            venue_offered[domain] = []
            provided_requestables[domain] = []
            domains_in_goal.append(domain)

        for t, sent_t in enumerate(dialog):
            for domain in goal.keys():
                # for computing success
                if '[' + domain + '_name]' in sent_t or '_id' in sent_t:
                    if domain in ['Delivery', 'Position', 'Assembly', 'Relocation']:

                        if domain in bs[t].keys():
                            state = bs[t][domain]
                        else:
                            state = {}

                        venues = self.db.queryResultVenues(domain, realDialogue['turn'][t * 2 + 1], state,
                                                           real_belief=False)

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[domain] = random.sample(venues, 1)
                        else:
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0] == ven:
                                    flag = True
                                    break
                            if not flag and venues:  # sometimes there are no results so sample won't work
                                # print venues
                                venue_offered[domain] = random.sample(venues, 1)
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[' + domain + '_name]'
                else:
                        if domain + '_' + requestable + ']' in sent_t:
                            provided_requestables[domain].append(requestable)

        # if name was given in the task
        for domain in goal.keys():
            # if name was provided for the user, the match is being done automatically
            # if realDialogue['goal'][domain].has_key('info'):
            if 'info' in realDialogue['goal'][domain]:
                # if realDialogue['goal'][domain]['info'].has_key('name'):
                if 'name' in realDialogue['goal'][domain]['info']:
                    venue_offered[domain] = '[' + domain + '_name]'
    """
        Given all inform and requestable slots
        we go through each domain from the user goal
        and check whether right entity was provided and
        all requestable slots were given to the user.
        The dialogue is successful if that's the case for all domains.
    """
        # HARD EVAL
        stats = {'Delivery': [0, 0, 0], 'Position': [0, 0, 0], 'Assembly': [0, 0, 0],
                 'Relocation': [0, 0, 0]}

        match = 0
        success = 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['Delivery', 'Position', 'Assembly', 'Relocation']:
                goal_venues = self.db.queryResultVenues(domain, goal[domain]['informable'], real_belief=True)
                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                    match += 1
                    match_stat = 1
            else:
                if domain + '_name]' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if soft_acc:
            match = float(match) / len(goal.keys())
        else:
            if match == len(goal.keys()):
                match = 1.0
            else:
                match = 0.0

        # SUCCESS
        if match == 1.0:
            for domain in domains_in_goal:
                success_stat = 0
                domain_success = 0
                if len(real_requestables[domain]) == 0:
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if soft_acc:
                success = float(success) / len(real_requestables)
            else:
                if success >= len(real_requestables):
                    success = 1
                else:
                    success = 0

        # print requests, 'DIFF', requests_real, 'SUCC', success
        return success, match, stats

    def _evaluateRealDialogue(self, dialog, filename):
        """Evaluation of the real dialogue.
        First we loads the user goal and then go through the dialogue history.
        Similar to evaluateGeneratedDialogue above."""
        domains = ['Delivery', 'Position', 'Assembly', 'Relocation']
        requestables = ['producttype', 'area', 'location']

        # get the list of domains in the goal
        domains_in_goal = []
        goal = {}
        for domain in domains:
            if dialog['turn'][domain]:
                turn = self._parseGoal(turn, dialog, domain)
                domains_in_goal.append(domain)

        # compute corpus success
        real_requestables = {}
        provided_requestables = {}
        venue_offered = {}
        for domain in turn.keys():
            provided_requestables[domain] = []
            venue_offered[domain] = []
            real_requestables[domain] = turn[domain]['requestable']

        # iterate each turn
        m_targetutt = [turn['text'] for idx, turn in enumerate(dialog['turn']) if idx % 2 == 1]
        for t in range(len(m_targetutt)):
            for domain in domains_in_goal:
                sent_t = m_targetutt[t]
                # for computing match - where there are limited entities
                if domain + '_name' in sent_t or '_id' in sent_t:
                    if domain in ['Delivery', 'Position', 'Assembly', 'Relocation']:
                        # HERE YOU CAN PUT YOUR BELIEF STATE ESTIMATION
                        venues = self.db.queryResultVenues(domain, dialog['turn'][t * 2 + 1])

                        # if venue has changed
                        if len(venue_offered[domain]) == 0 and venues:
                            venue_offered[domain] = random.sample(venues, 1)
                        else:
                            flag = False
                            for ven in venues:
                                if venue_offered[domain][0] == ven:
                                    flag = True
                                    break
                            if not flag and venues:  # sometimes there are no results so sample won't work
                                # print venues
                                venue_offered[domain] = random.sample(venues, 1)
                    else:  # not limited so we can provide one
                        venue_offered[domain] = '[' + domain + '_name]'

                for requestable in requestables:
                    if domain + '_' + requestable in sent_t:
                        provided_requestables[domain].append(requestable)

        # HARD (0-1) EVAL
        stats = {'Delivery': [0, 0, 0], 'Position': [0, 0, 0], 'Assembly': [0, 0, 0],
                 'Relocation': [0, 0, 0]}

        match, success = 0, 0
        # MATCH
        for domain in goal.keys():
            match_stat = 0
            if domain in ['Delivery', 'Position', 'Assembly', 'Relocation']:
                goal_venues = self.db.queryResultVenues(domain, dialog['turn'][domain]['search_result'], real_belief=True)
                # print(goal_venues)
                if type(venue_offered[domain]) is str and '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1
                elif len(venue_offered[domain]) > 0 and venue_offered[domain][0] in goal_venues:
                    match += 1
                    match_stat = 1

            else:
                if domain + '_name' in venue_offered[domain]:
                    match += 1
                    match_stat = 1

            stats[domain][0] = match_stat
            stats[domain][2] = 1

        if match == len(goal.keys()):
            match = 1
        else:
            match = 0

        # SUCCESS
        if match:
            for domain in domains_in_goal:
                domain_success = 0
                success_stat = 0
                if len(real_requestables[domain]) == 0:
                    # check that
                    success += 1
                    success_stat = 1
                    stats[domain][1] = success_stat
                    continue
                # if values in sentences are super set of requestables
                for request in set(provided_requestables[domain]):
                    if request in real_requestables[domain]:
                        domain_success += 1

                if domain_success >= len(real_requestables[domain]):
                    success += 1
                    success_stat = 1

                stats[domain][1] = success_stat

            # final eval
            if success >= len(real_requestables):
                success = 1
            else:
                success = 0

        return goal, success, match, real_requestables, stats

    def _parse_entities(self, tokens):
        entities = []
        for t in tokens:
            if '[' in t and ']' in t:
                entities.append(t)
        return entities


    def evaluateModel(self, dialogues, dialogues_bs, real_dialogues=False, mode='valid'):
        """Gathers statistics for the whole sets."""
        delex_dialogues = self.delex_dialogues
        successes, matches = 0, 0
        total = 0

        gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                     'taxi': [0, 0, 0],
                     'hospital': [0, 0, 0], 'police': [0, 0, 0]}
        sng_gen_stats = {'restaurant': [0, 0, 0], 'hotel': [0, 0, 0], 'attraction': [0, 0, 0], 'train': [0, 0, 0],
                         'taxi': [0, 0, 0], 'hospital': [0, 0, 0], 'police': [0, 0, 0]}
        all_slots = 0
        hit_slots = 0
        for filename, dial in dialogues.items():

            filename = filename.upper().split('.')[0]
            filename = filename + '.json'

            try:

                data = delex_dialogues[filename]
                bs_state = dialogues_bs[filename.split('.')[0]]
            except Exception:
                print(filename)
                continue

            goal, success, match, requestables, _ = self._evaluateRealDialogue(data, filename)
            success, match, stats = self._evaluateGeneratedDialogue((dial, bs_state), goal, data, requestables,
                                                                    soft_acc=mode == 'soft')

            successes += success
            matches += match
            total += 1

            for domain in gen_stats.keys():
                gen_stats[domain][0] += stats[domain][0]
                gen_stats[domain][1] += stats[domain][1]
                gen_stats[domain][2] += stats[domain][2]

            if 'SNG' in filename:
                for domain in gen_stats.keys():
                    sng_gen_stats[domain][0] += stats[domain][0]
                    sng_gen_stats[domain][1] += stats[domain][1]
                    sng_gen_stats[domain][2] += stats[domain][2]
        print(hit_slots, all_slots)
        if real_dialogues:
            # BLUE SCORE
            corpus = []
            model_corpus = []
            bscorer = BLEUScorer()

            for dialogue in dialogues.keys():

                data = delex_dialogues[dialogue.split('.')[0].upper() + '.json']
                model_turns, corpus_turns = [], []
                sys_turns = data['turn']
                for idx, turn in enumerate(sys_turns):
                    if idx % 2 == 1:
                        corpus_turns.append([turn['system']])
                for turn in dialogues[dialogue]:
                    model_turns.append([turn])

                if len(model_turns) == len(corpus_turns):
                    corpus.extend(corpus_turns)
                    model_corpus.extend(model_turns)
                else:
                    raise ('Wrong amount of turns')
            blue_score = bscorer.score(model_corpus, corpus)
            smooth = SmoothingFunction()
            corpus_ = [[i[0]] for i in corpus]
            hypothesis_ = [i[0] for i in model_corpus]

            print('corpus level', corpus_bleu(corpus_, hypothesis_, smoothing_function=smooth.method1))
        else:
            blue_score = 0.

        report = ""
        report += '{} Corpus Matches : {:2.2f}%'.format(mode, (matches / float(total) * 100)) + "\n"
        report += '{} Corpus Success : {:2.2f}%'.format(mode, (successes / float(total) * 100)) + "\n"
        report += '{} Corpus BLEU : {:2.2f}%'.format(mode, blue_score) + "\n"
        report += 'Total number of dialogues: %s ' % total

        print(report)
        combined = (successes / float(total) + matches / float(total)) / 2 + blue_score
        print(f'Combined Score {combined}')

        return report, successes / float(total), matches / float(total)


from parser import parse_decoding_results
import glob
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", default=None, type=str, required=True, help="The input evaluation file.")
    parser.add_argument("--eval_mode", default='test', type=str, help="valid/test")

    args = parser.parse_args()

    mode = "test"
    with open(cfg.dataset_path_test_file, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    evaluator = IRWOZEvaluator(mode,lines)


    res, res_bs = parse_decoding_results(cfg.dataset_path_decoded_file, mode)

    # PROVIDE HERE YOUR GENERATED DIALOGUES INSTEAD
    generated_data = res
    generated_proc_belief = res_bs
    evaluator.evaluteDST(generated_data, generated_proc_belief, True, mode=mode)
    evaluator.evaluateModel(generated_data, generated_proc_belief, True, mode=mode)


