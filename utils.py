import numpy as np
from data import compute_f1


def calculate_acc(tag2idx):
    ## calc metric
    y_true = np.array([tag2idx[line.split()[1]] for line in open('result', 'r').read().splitlines() if len(line) > 0])
    y_pred = np.array([tag2idx[line.split()[2]] for line in open('result', 'r').read().splitlines() if len(line) > 0])

    acc = (y_true == y_pred).astype(np.int32).sum() / len(y_true)

    print("acc=%.4f" % acc)


def calculate_f1(tag2idx, idx2tag):
    y_true = []
    y_pred = []
    with open('result', 'r', encoding='UTF-8') as f:
        t_sentence = []
        p_sentence = []
        for line in f:
            line = line.strip()
            if len(line) != 0:
                t_sentence.append(tag2idx[line.split()[1]])
                p_sentence.append(tag2idx[line.split()[2]])
            elif len(t_sentence) != 0:
                y_true.append(t_sentence)
                y_pred.append(p_sentence)

                t_sentence = []
                p_sentence = []

    pre, rec, f1 = compute_f1(y_pred, y_true, idx2tag, 'O')
    pre_b, rec_b, f1_b = compute_f1(y_pred, y_true, idx2tag, 'B')

    if f1_b > f1:
        print("Setting wrong tags to B- improves from %.4f to %.4f" % (f1, f1_b))
        pre, rec, f1 = pre_b, rec_b, f1_b

    print('precision={}, recall={}, f1={}'.format(pre, rec, f1))
