import math
from collections.abc import Iterable

import numpy as np
import sacrebleu
import sklearn.metrics
import random
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch


def mean(arr):
    return sum(arr) / len(arr)


def pop_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / len(arr))


def sample_stddev(arr):
    mu = mean(arr)
    return math.sqrt(sum([(x - mu) ** 2 for x in arr]) / (len(arr) - 1))


def mean_stderr(arr):
    return sample_stddev(arr) / math.sqrt(len(arr))


def median(arr):
    return arr[len(arr) // 2]


def matthews_corrcoef(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    return sklearn.metrics.matthews_corrcoef(golds, preds)


def f1_score(items):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = sklearn.metrics.f1_score(golds, preds)

    return np.max(fscore)


def acc_all(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        paragraph_id = doc["idx"]["paragraph"]
        question_id = doc["idx"]["question"]
        if (paragraph_id, question_id) not in question_scoring_dict:
            question_scoring_dict[(paragraph_id, question_id)] = []

        gold_label = doc["label"] == 1

        question_scoring_dict[(paragraph_id, question_id)].append(gold_label == pred)
    acc = np.mean([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def acc_all_stderr(items):
    # Only count as correct if all answers are labeled correctly for each question
    question_scoring_dict = {}
    preds = list(zip(*items))[0]
    docs = list(zip(*items))[1]

    for doc, pred in zip(docs, preds):
        question_id = doc["idx"]["question"]
        if question_id not in question_scoring_dict:
            question_scoring_dict[question_id] = []

        gold_label = doc["label"] == 1
        question_scoring_dict[question_id].append(gold_label == pred)

    acc = mean_stderr([int(all(x)) for x in question_scoring_dict.values()])
    return acc


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Compute max metric between prediction and each ground truth."""
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def perplexity(items):
    return math.exp(-mean(items))


def weighted_mean(items):
    a, b = zip(*items)
    return sum(a) / sum(b)


def weighted_perplexity(items):
    return math.exp(-weighted_mean(items))


def bits_per_byte(items):
    return -weighted_mean(items) / math.log(2)


def bleu(items):
    """The Bilingual Evaluation Understudy Score, or BLEU for short, is a metric
    for evaluating a generated sentence to a reference sentence. It counts matching
    n-grams in the candidate translation to n-grams in the reference text, where
    1-gram or unigram would be each token and a bigram comparison would be each
    word pair. The comparison is made regardless of word order
    Source: https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
    Paper: https://www.aclweb.org/anthology/P02-1040/

    Higher is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_bleu(preds, refs).score


def chrf(items):
    """chrF++ is a tool for automatic evaluation of machine translation output
    based on character n-gram precision and recall enhanced with word n-grams.
    Source: https://github.com/m-popovic/chrF
    Paper: https://www.aclweb.org/anthology/W15-3049.pdf

    Higher is better  # TODO I think
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_chrf(preds, refs).score


def ter(items):
    """Translation Error Rate is an error metric for machine translation that
    measures the number of edits required to change a system output into one
    of the references
    Source: http://www.cs.umd.edu/~snover/tercom/
    Paper: http://mt-archive.info/AMTA-2006-Snover.pdf

    Lower is better
    """
    refs = list(zip(*items))[0]
    preds = list(zip(*items))[1]
    refs, preds = _sacreformat(refs, preds)
    return sacrebleu.corpus_ter(preds, refs).score


def is_non_str_iterable(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)


def _sacreformat(refs, preds):
    """Format refs and preds for sacrebleu corpus calculation. It is very particular"""
    # Sacrebleu expects (List[str], List[List[str])
    #   e.g. sacrebleu.corpus_bleu([pred_t], [[ref1_stream], [ref2_stream], ...])

    # Note [ref1_stream] is the first reference for each pred.
    # So lists are size N and (M, N) for N preds and M possible refs for each pred
    # This is a different order of dimensions that I would expect

    # We expect refs to be List[str] or List[List[str]], the outer list corresponding to preds
    # Must become List[List[str]] with the inner list corresponding to preds
    if not is_non_str_iterable(refs):
        refs = list(refs)
    if not is_non_str_iterable(refs[0]):
        refs = [[ref] for ref in refs]
    refs = list(zip(*refs))
    # Note the number of refs in each ref list much match the number of preds

    # We expect preds to be List[str] or List[List[str]]. Must become List[str]
    if not is_non_str_iterable(preds):
        preds = list(preds)
    if is_non_str_iterable(preds[0]):
        assert len(preds[0]) == 1, f"Pred must be a str, was {preds[0]}"
        preds = [pred[0] for pred in preds]

    return refs, preds


# stderr stuff


class _bootstrap_internal:
    def __init__(self, f, n):
        self.f = f
        self.n = n

    def __call__(self, v):
        i, xs = v
        rnd = random.Random()
        rnd.seed(i)
        res = []
        for _ in range(self.n):
            res.append(self.f(rnd.choices(xs, k=len(xs))))
        return res


def bootstrap_stderr(f, xs, iters):
    import multiprocessing as mp

    pool = mp.Pool(mp.cpu_count())
    # this gives a biased estimate of the stderr (i.e w/ the mean, it gives something
    # equivalent to stderr calculated without Bessel's correction in the stddev.
    # Unfortunately, I haven't been able to figure out what the right correction is
    # to make the bootstrap unbiased - i considered multiplying by sqrt(n/(n-1)) but
    # that would be ad-hoc and I can't prove that that would actually be an unbiased estimator)
    # Thankfully, shouldn't matter because our samples are pretty big usually anyways
    res = []
    chunk_size = min(1000, iters)
    from tqdm import tqdm

    print("bootstrapping for stddev:", f.__name__)
    for bootstrap in tqdm(
        pool.imap(
            _bootstrap_internal(f, chunk_size),
            [(i, xs) for i in range(iters // chunk_size)],
        ),
        total=iters // chunk_size,
    ):
        # sample w replacement
        res.extend(bootstrap)

    pool.close()
    return sample_stddev(res)


def stderr_for_metric(metric, bootstrap_iters):
    bootstrappable = [
        median,
        matthews_corrcoef,
        f1_score,
        perplexity,
        bleu,
        chrf,
        ter,
    ]

    if metric in bootstrappable:
        return lambda x: bootstrap_stderr(metric, x, iters=bootstrap_iters)

    stderr = {mean: mean_stderr, acc_all: acc_all_stderr}

    return stderr.get(metric, None)

def get_perplexity_of_one_sentence(model, sentence, window=5):
    loss_fct = CrossEntropyLoss()
    stride = 1
    tokenizer = model.tokenizer
    encoded = torch.unsqueeze(torch.Tensor(tokenizer.encode(sentence, return_tensors="pt")), dim=0).to("cuda").long()  # make it batched
    seq_len = encoded.shape[1]
    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + window, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encoded[:, begin_loc:end_loc].to("cuda")
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100


        with torch.no_grad():
            outputs = model._model_call(input_ids)

            ##  calculating loss

            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.

            neg_log_likelihood = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl

def autoregressive_for_choices(model, tokenizer, c, q, cs):
    full_context = c + " " + q + " "
    log_prob_list = []

    for choice in cs:
        log_prob_sum = 0
        input_ids = torch.unsqueeze(torch.Tensor(tokenizer.encode(full_context, return_tensors='pt')), dim=0).to("cuda").long()
        output = model(input_ids)
        for word in tokenizer.encode(choice, add_special_tokens=False):
            to_be_concate = torch.unsqueeze(torch.Tensor([word]), dim=0).to("cuda").long()

            next_word_logits = output[0, -1, :]

            next_word_probs = torch.softmax(next_word_logits, dim=-1)

            next_word_log_prob = torch.log(next_word_probs[word])

            log_prob_sum += next_word_log_prob.item()

            input_ids = torch.cat([input_ids, to_be_concate], dim=1).to("cuda")

            output = model(input_ids) 

        log_prob_list.append(log_prob_sum)

    final_choice = np.argmax(log_prob_list)

    return final_choice
        

def autoregressive_for_choices_1(model, tokenizer, c, q, cs):
    full_context = c + " " + q + " "
    log_prob_list = []

    for choice in cs:
        log_prob_sum = 0
        input_ids = torch.Tensor(tokenizer.encode(full_context, return_tensors='pt')).to("cuda").long()
        output = model(input_ids).logits
        for word in tokenizer.encode(choice, add_special_tokens=False):
            to_be_concate = torch.unsqueeze(torch.Tensor([word]), dim=0).to("cuda").long()

            next_word_logits = output[0, -1, :]

            next_word_probs = torch.softmax(next_word_logits, dim=-1)

            next_word_log_prob = torch.log(next_word_probs[word])

            log_prob_sum += next_word_log_prob.item()

            input_ids = torch.cat([input_ids, to_be_concate], dim=1).to("cuda")

            output = model(input_ids).logits

        log_prob_list.append(log_prob_sum)

    final_choice = np.argmax(log_prob_list)

    return final_choice

def yesno(x):
    if x:
        return "yes"
    else:
        return "no"
