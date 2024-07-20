import os, json

golds = []

with open('/data/hzh22/CodeReviewProcesser/postprocess/data2/ref-test.jsonl', "r") as f:
    for line in f:
        tgt_txt = 'bad' if json.loads(line)['y'] == 1 else 'good'
        golds.append(tgt_txt)


with open(os.path.join('/data/hzh22/CodeBERT/CodeReviewer/code/sh/els_comment_model_onediff', "golds.txt"), "w", encoding="utf-8") as f:
    for gold in golds:
        f.write(gold.strip() + "\n")   