from sklearn.metrics import classification_report
def read_data(file_path):
    res = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            res.append(line.split('\t')[1].replace('\n', '').replace('bad', 'inconsistent').replace('good', 'consistent').split(' '))
    return res

# test CodeReviewer output
def read_data2(file_path):
    res = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            res.append(line.replace('\n', '').replace('bad', 'inconsistent').replace('good', 'consistent').split(' '))
            
    return res


dev_output = read_data("/data/hzh22/pt4code/PT4Code/defect/prompt/MCC/mcc_model_without_comment/test_0.output")
dev_gold = read_data("/data/hzh22/pt4code/PT4Code/defect/prompt/MCC/mcc_model_without_comment/test_0.gold")
# codeT5 without comment
test0_output = read_data("/data/hzh22/pt4code/PT4Code/defect/prompt/MCC/mcc_model_without_comment/test_0.output")
test0_gold = read_data("/data/hzh22/pt4code/PT4Code/defect/prompt/MCC/mcc_model_without_comment/test_0.gold")

# CodeReviewer code type
test_output = read_data2("/data/hzh22/CodeBERT/CodeReviewer/code/sh/els_comment_model_onediff/checkpoints-last-6.0/preds.txt")
test_gold = read_data2("/data/hzh22/CodeBERT/CodeReviewer/code/sh/els_comment_model_onediff/checkpoints-last-6.0/golds.txt")

print("#### Testset:")
print(classification_report(test_gold, test_output, digits=4))


def calc_score(gold, output):
    results = []
    c = []
    ic = []
    for i in range(len(gold)):
        g, o = gold[i], output[i]
        if o == 'bad':
            if g == 'good':
                ic.append(0)
            else:
                ic.append(1)
        else:
            if g == 'good':
                c.append(1)
            else:
                c.append(0)
        p = len(set(g) & set(o)) / len(set(o))
        r = len(set(g) & set(o)) / len(set(g))
        f1 = 2 * p * r / (p + r) if p and r else 0
        em = 1 if g == o else 0
        results.append([p, r, f1, em])
    precision = sum([res[0] for res in results]) / len(results)
    recall = sum([res[1] for res in results]) / len(results)
    f1_score = sum([res[2] for res in results]) / len(results)
    em_acc = sum([res[3] for res in results]) / len(results)
    return precision, recall, f1_score, em_acc
'''
p, r, f1, em_acc = calc_score(dev_gold, dev_output)
f1 = 2 * p * r / (p + r)
print("#### Dev:")
print("Precision:{}".format(p))
print("Recall:{}".format(r))
print("F1-Score:{}".format(f1))
print("EM Acc:{}".format(em_acc))


print("#### Testset:")
p, r, f1, em_acc = calc_score(test0_gold, test0_output)
f1 = 2 * p * r / (p + r)
print("Precision:{}".format(p))
print("Recall:{}".format(r))
print("F1-Score:{}".format(f1))
print("EM Acc:{}".format(em_acc))
'''