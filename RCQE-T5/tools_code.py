import json
import time

from openprompt.data_utils import InputExample


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)



class InputFeatures(object):
    """A single training/test features for an example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    # collect texts
    codes = []
    target_nl = []
    for example_id, example in enumerate(examples):
        codes.append(example.source)

        if stage == "test":
            target_nl.append("None")
        else:
            target_nl.append(example.target)

    # begin tokenizing
    encoded_codes = tokenizer(
        codes, padding=True, verbose=False, add_special_tokens=True,
        truncation=True, max_length=args.max_source_length, return_tensors='pt')

    encoded_targets = tokenizer(
        target_nl, padding=True, verbose=False, add_special_tokens=True,
        truncation=True, max_length=args.max_source_length, return_tensors='pt')

    return {'source_ids':encoded_codes['input_ids'], 'target_ids':encoded_targets['input_ids'],
            'source_mask':encoded_codes['attention_mask'], 'target_mask':encoded_targets['attention_mask']}


def read_prompt_examples(file_type):
    """Read examples from filename."""
    examples = []

    file_name = '../data/ref-{}.jsonl'.format(file_type)
    with open(file_name, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            old = js["old"].split("\n")[1:] 
            new = js["new"].split("\n")[1:] 
            old_diff = "".join(old)
            new_diff = "".join(new)
            if 'idx' not in js:
                js['idx'] = idx
            tgt_txt = 'bad' if js['y'] == 1 else 'good'
            result = "Old Code: " + old_diff 
            newCode =  "New Code: " + new_diff
            comment = js["comment"]
            example = InputExample(guid=idx, text_a=result, text_b=newCode, tgt_text=tgt_txt)
            # code = ' '.join(js['code_tokens']).replace('\n', ' ')
            # code = ' '.join(code.strip().split())
            # nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            # nl = ' '.join(nl.strip().split())

            examples.append(example)
    print("len examples")
    print(len(examples))
    return examples
    # with open(file_name, encoding="utf-8") as f:
    #     for idx, line in enumerate(f):
    #         content = line.strip().split('\t')
    #         consistent = True if content[0] == '0' else False
    #         method_name = content[1]
    #         method_context = content[2]
    #         tgt_txt = "good" if consistent else "bad"
    #         example = InputExample(guid=idx, text_a=method_context, text_b=method_name, tgt_text=tgt_txt)
    #         # code = ' '.join(js['code_tokens']).replace('\n', ' ')
    #         # code = ' '.join(code.strip().split())
    #         # nl = ' '.join(js['docstring_tokens']).replace('\n', '')
    #         # nl = ' '.join(nl.strip().split())

    #         examples.append(example)

    # return examples