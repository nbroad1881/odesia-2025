import re
import pickle
import hydra
from pathlib import Path
from transformers import AutoTokenizer
import pandas as pd

import numpy as np

def post_process_diann(tokens, preds, tokenizer, first_split):
    pred_str = (
        tokenizer.decode(preds)[0].split(first_split)[1].split(tokenizer.eos_token)[0]
    )
    pred_list = pred_str.split("\n")

    pred_idx = 0
    inner_idx = 0
    pred_labels = []
    in_entity = False

    for t in tokens:
        pred_tokens = pred_list[pred_idx].split()
        if pred_tokens[inner_idx] == t:
            if not in_entity:
                pred_labels.append("B-DIS")
                in_entity = True
            else:
                pred_labels.append("I-DIS")
            inner_idx += 1
        else:
            pred_labels.append("O")
            if in_entity:
                pred_idx += 1
                in_entity = False

            inner_idx = 0

    pred_str = [t for t in pred_str if t in tokens]
    return pred_str


def post_process_diann2(tokens, preds, tokenizer, first_split):

    text = " ".join(tokens)

    char_list = list(text)
    label_list = ["O"] * len(char_list)
    token_idx_list = np.array([-1] * len(char_list))

    start_idx = 0
    for i, t in enumerate(tokens):
        token_idx_list[start_idx : start_idx + len(t)] = i
        start_idx += len(t) + 1

    for pred in pred_list:

        pred_words = pred.split()

        for m in re.finditer(re.escape(pred), text):
            start, end = m.start(), m.end()
            for i in range(start, start + len(pred_words[0])):
                label_list[i] = "B-DIS"
            for i in range(start + len(pred_words[0]), end):
                label_list[i] = "I-DIS"

    final_labels = ["O"] * len(tokens)

    token_idx_list = token_idx_list.tolist()

    for i, t in enumerate(tokens):
        if t == "":
            continue

        first_idx = token_idx_list.index(i)
        final_labels[i] = label_list[first_idx]

    return final_labels


def post_process_dipromats(pred_str_list):

    def process_single(p):
        letter = p.strip()[0]

        if letter == "P":
            t1 = "false"
            t2 = ["false"]
            t3 = ["false"]
            return t1, t2, t3

        letters = [x.strip()[0] for x in p.strip().split("\n")]

        t1 = "true"
        t2 = []
        t3 = []

        if "A" in letters:
            t2.append("1 appeal to commonality")
        if any(x in letters for x in "BCDEFGHIJKL"):
            t2.append("2 discrediting the opponent")
        if "M" in letters:
            t2.append("3 loaded language")
        if "N" in letters or "O" in letters:
            t2.append("4 appeal to authority")

        preds = p.strip().split("\n")

        for pred in preds:
            if "appeal to commonality" in pred:
                t3.append("1 " + pred[1:].strip())
            elif "discrediting the opponent" in pred:
                if pred[0] == "H":
                    t3.append("2 discrediting the opponent - fear appeals (destructive)")
                else:
                    t3.append("2 " + pred[1:].strip())
            elif "loaded language" in pred:
                t3.append("3 " + pred[1:].strip())
            elif "appeal to authority" in pred:
                t3.append("4 " + pred[1:].strip())

        return t1, t2, t3

    return zip(*[process_single(p) for p in pred_str_list])


def post_process_exist_2022(all_probs):
    return post_process_exist_2022_t1(all_probs), post_process_exist_2022_t2(all_probs)

def post_process_exist_2022_t1(all_probs):
    """
        Probs shoudl be a list of lists of 6 values representing the scores for:
        [
        "ideological inequality",
        "stereotyping dominance",
        "sexual violence",
        "misogyny and/or non-sexual violence",
        "objectification",
        "not sexist",
    ]
    """

    results = []

    for probs in all_probs:

        not_sexist_prob = probs[-1]

        if not_sexist_prob > 0.5:
            results.append("non-sexist")
        else:
            results.append("sexist")

    return results

def post_process_exist_2022_t2(all_probs):
    """
        Probs shoudl be a list of lists of 6 values representing the scores for:
        [
        "ideological inequality",
        "stereotyping dominance",
        "sexual violence",
        "misogyny and/or non-sexual violence",
        "objectification",
        "not sexist",
    ]
    """

    idx2label = [
        "ideological-inequality",
        "stereotyping-dominance",
        "sexual-violence",
        "misogyny-non-sexual-violence",
        "objectification",
        "non-sexist",
    ]

    results = []

    preds = np.array(all_probs).argmax(-1)

    for pred in preds:
        results.append(idx2label[pred])

    return results

def post_process_exist_2023_t1(all_probs):
    """
    1. direct
    2. reported
    3. judgmental
    4. not sexist
    """

    not_sexist_probs = np.array(all_probs)[:, :-1]

    return [
        {"YES": 1 - x, "NO": x}
        for x in not_sexist_probs
    ]

def post_process_exist_2023_t2(all_probs):
    """
    1. direct
    2. reported
    3. judgmental
    4. not sexist
    """

    return [
        {"DIRECTED": x[0], "REPORTED": x[1], "JUDGMENTAL": x[2], "NO": x[3]}
        for x in all_probs
    ]

def post_process_exist_2023_t3(pred_str_list):
    """
    """

    idx2label = [
        "ideological-inequality",
        "stereotyping-dominance",
        "sexual-violence",
        "misogyny-non-sexual-violence",
        "objectification",
        "non-sexist",
    ]

    def assign_prob(s):
        scores = {l: 0 for l in idx2label}
        if "ideological" in s:
            scores[idx2label[0]] = 1
        elif "stereotyping" in s:
            scores[idx2label[1]] = 1
        elif "sexual violence" in s:
            scores[idx2label[2]] = 1
        elif "misogyny" in s:
            scores[idx2label[3]] = 1
        elif "objectification" in s:
            scores[idx2label[4]] = 1
        else:
            scores[idx2label[5]] = 1

    

    return [
       assign_prob(x)
        for x in pred_str_list
    ]

def post_process_sqac_squad_2024_t1(pred_str_list):
    return pred_str_list


def route(task):
    if "exist_2022" in task:
        return post_process_exist_2022
    elif "exist_2023" in task and "t1" in task:
        return post_process_exist_2023_t1
    elif "exist_2023" in task and "t2" in task:
        return post_process_exist_2023_t2
    elif "exist_2023" in task:
        return post_process_exist_2023_t3
    elif "sqac_squad_2024" in task:
        return post_process_sqac_squad_2024_t1
    elif "dipromats" in task:
        return post_process_dipromats
    elif "diann" in task:
        return post_process_diann2
        

@hydra.main(config_path="conf", config_name="ministral_v1.yaml")
def main(cfg):

    output_dir = Path(cfg.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    preds_dir = Path(cfg.preds_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)

    for task in cfg.tasks:
        with open(preds_dir / f"{task.replace('/', '_')}_preds.pkl", "rb") as f:
            preds = pickle.load(f)

        if ("exist_2022" in task) or ("exist_2023" in task and "t3" not in task):
            probs, gt = preds

            results = route(task)(probs)

            if "exist_2022" in task:
                t1_results, t2_results = results

                pred_df = pd.DataFrame({"t1_pred": t1_results, "t2_pred": t2_results, "gt": gt})

            else:
                pred_df = pd.DataFrame({"pred": results, "gt": gt})

            pred_df.to_parquet(
                output_dir / f"{task.replace('/', '_')}_preds_df.parquet", index=False
            )
        else:
            pred_ids, gt = preds

            pred_str = tokenizer.batch_decode(pred_ids)
            pred_str = [
                x.split(cfg.split_one)[-1].split(cfg.eos_token)[0] for x in pred_str
            ]

            if "diann" in task:
                continue

            if "dipromats" in task:
                t1, t2, t3 = post_process_dipromats(pred_str)

                pred_df = pd.DataFrame({"t1_pred": t1, "t2_pred": t2, "t3_pred": t3, "gt": gt})

            if "sqac" in task:
                pred_df = pd.DataFrame({"pred": pred_str, "gt": gt})
            
            if "exist" in task:
                pred_df = pd.DataFrame({"pred": post_process_exist_2023_t3(pred_str), "gt": gt})

            pred_df.to_parquet(
                output_dir / f"{task.replace('/', '_')}_preds_df.parquet", index=False
            )


if __name__ == "__main__":
    main()
