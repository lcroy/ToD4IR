import math
import re
from collections import Counter
from config import Config

cfg = Config()

from nltk.util import ngrams

timepat = re.compile("\d{1,2}[:]\d{1,2}")
pricepat = re.compile("\d{1,3}[.]\d{1,2}")


file=open
fin = open(cfg.pair_path)
replacements = []
for line in fin.readlines():
    tok_from, tok_to = line.replace('\n', '').split('\t')
    replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text


def normalize(text):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\":\<>@\(\)]', '', text)

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text


class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, hypothesis, corpus, n=1):
        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in zip(hypothesis, corpus):
            if type(hyps[0]) is list:
                hyps = [hyp.split() for hyp in hyps[0]]
            else:
                hyps = [hyp.split() for hyp in hyps]

            refs = [ref.split() for ref in refs]

            # Shawn's evaluation
            refs[0] = [u'GO_'] + refs[0] + [u'EOS_']
            hyps[0] = [u'GO_'] + hyps[0] + [u'EOS_']

            for idx, hyp in enumerate(hyps):
                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
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


class GentScorer(object):
    def __init__(self, detectfile):
        self.bleuscorer = BLEUScorer()

    def scoreBLEU(self, parallel_corpus):
        return self.bleuscorer.score(parallel_corpus)


def sentence_bleu_4(hyp, refs, weights=[0.25, 0.25, 0.25, 0.25]):
    # input : single sentence, multiple references
    count = [0, 0, 0, 0]
    clip_count = [0, 0, 0, 0]
    r = 0
    c = 0

    for i in range(4):
        hypcnts = Counter(ngrams(hyp, i + 1))
        cnt = sum(hypcnts.values())
        count[i] += cnt

        # compute clipped counts
        max_counts = {}
        for ref in refs:
            refcnts = Counter(ngrams(ref, i + 1))
            for ng in hypcnts:
                max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
        clipcnt = dict((ng, min(count, max_counts[ng])) \
                       for ng, count in hypcnts.items())
        clip_count[i] += sum(clipcnt.values())

    bestmatch = [1000, 1000]
    for ref in refs:
        if bestmatch[0] == 0:
            break
        diff = abs(len(ref) - len(hyp))
        if diff < bestmatch[0]:
            bestmatch[0] = diff
            bestmatch[1] = len(ref)
    r = bestmatch[1]
    c = len(hyp)

    p0 = 1e-7
    bp = math.exp(-abs(1.0 - float(r) / float(c + p0)))

    p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 for i in range(4)]
    s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)
    bleu_hyp = bp * math.exp(s)

    return bleu_hyp