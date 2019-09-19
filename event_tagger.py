# -*- coding: utf-8 -*-

import os
import tqdm
import torch
import torch.nn as nn
from torch.utils import data
from pytorch_transformers import BertTokenizer, BertModel, AdamW, WarmupLinearSchedule
from data import EventDataset, read_event_data, pad
from utils import calculate_acc, calculate_f1
from model import Net, train, eval


def event_tagger():
    # Read event data
    en_train = read_event_data('en/train.txt')
    en_dev = read_event_data('en/dev.txt')
    en_test = read_event_data('en/test.txt')

    it_train = read_event_data('it/train.txt')
    it_dev = read_event_data('it/dev.txt')
    it_test = read_event_data('it/test.txt')

    print('English TimeML:', len(en_train), len(en_dev), len(en_test))
    print('Italian News:', len(it_train), len(it_dev), len(it_test))

    tags = list(set(word_label[1] for sent in it_train for word_label in sent))
    print(len(tags))

    # By convention, the 0'th slot is reserved for padding.
    tags = ["<pad>"] + tags

    tag2idx = {tag: idx for idx, tag in enumerate(tags)}
    idx2tag = {idx: tag for idx, tag in enumerate(tags)}

    print(tag2idx)
    print(idx2tag)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

    model = Net(vocab_size=len(tag2idx), device=device)
    model.to(device)
    model = nn.DataParallel(model)

    # One fine-tuning step
    train_dataset = EventDataset(en_train, tokenizer, tag2idx)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=8,
                                 shuffle=True,
                                 num_workers=1,
                                 collate_fn=pad)

    eval_dataset = EventDataset(it_test, tokenizer, tag2idx)

    test_iter = data.DataLoader(dataset=eval_dataset,
                                batch_size=8,
                                shuffle=False,
                                num_workers=1,
                                collate_fn=pad)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    num_epoch = 1
    base_lr = 0.001
    decay_factor = 0.2
    discriminative_fine_tuning = True
    gradual_unfreezing = False

    # params order top to bottom
    group_to_discriminate = ['classifier', 'bert']
    no_decay = ['bias', 'LayerNorm.weight']

    if discriminative_fine_tuning:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and not 'bert' in n],
             'layers': [n for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay) and not 'bert' in n],
             'lr': 0.001, 'name': 'classifier.decay', 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not 'bert' in n],
             'layers': [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not 'bert' in n],
             'lr': 0.001, 'name': 'classifier.no_decay', 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' in n],
             'layers': [n for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and 'bert' in n],
             'lr': 0.00002, 'name': 'bert.decay', 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'bert' in n],
             'layers': [n for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and 'bert' in n],
             'lr': 0.00002, 'name': 'bert.no_decay', 'weight_decay': 0.0}
        ]
    else:
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=len(train_iter) * num_epoch // 10,
                                     t_total=len(train_iter) * num_epoch)

    for e in range(num_epoch):
        unfreeze = (True, False)[e != 0]

        if discriminative_fine_tuning and gradual_unfreezing:
            for pg in optimizer.param_groups:
                layers = ''
                for layer in pg['layers']:
                    layers += layer + ';'
                # print('epoch: {}, Layers: {}'.format(e, layers))
                if 'bert' in pg['name']:
                    for param in pg['params']:
                        param.requires_grad = unfreeze

        loss = train(model, train_iter, optimizer, scheduler, criterion)
        acc = eval(model, test_iter, idx2tag)

        print("epoch: {}, loss: {}".format(e, loss))
        print("epoch: {}, acc: {}".format(e, acc))

    '''
    ## Second fine-tuning step (epoch=1)
    
    train_dataset = EventDataset(it_train, tokenizer, tag2idx)
    for e in range(num_epoch):
        unfreeze = (True, False)[e != 0]

        if discriminative_fine_tuning and gradual_unfreezing:
            for pg in optimizer.param_groups:
                layers = ''
                for layer in pg['layers']:
                    layers += layer + ';'
                # print('epoch: {}, Layers: {}'.format(e, layers))
                if 'bert' in pg['name']:
                    for param in pg['params']:
                        param.requires_grad = unfreeze

        loss = train(model, train_iter, optimizer, scheduler, criterion)
        acc = eval(model, test_iter, idx2tag)

        print("epoch: {}, loss: {}".format(e, loss))
        print("epoch: {}, acc: {}".format(e, acc))
    '''

    calculate_acc()
    calculate_f1()


if __name__ == "__main__":
    event_tagger()
