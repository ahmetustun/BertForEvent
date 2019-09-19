import tqdm
import torch
import torch.nn as nn
from pytorch_transformers import BertModel


# from https://github.com/Kyubyong/nlp_made_easy
class Net(nn.Module):
    def __init__(self, vocab_size=None, device='cpu'):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')

        self.fc = nn.Linear(768, vocab_size)
        self.device = device

    def forward(self, x, y):
        '''
        x: (N, T). int64
        y: (N, T). int64
        '''
        x = x.to(self.device)
        y = y.to(self.device)

        if self.training:
            self.bert.train()
            enc, _ = self.bert(x)
        else:
            self.bert.eval()
            with torch.no_grad():
                enc, _ = self.bert(x)

        logits = self.fc(enc)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat


def train(model, iterator, optimizer, scheduler, criterion):
    loss_total = 0.0
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        _y = y  # for monitoring
        logits, y, _ = model(x, y)  # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1])  # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        scheduler.step()
        optimizer.step()
        model.zero_grad()

        loss_total += loss.item()
        return loss_total


def eval(model, iterator, idx2tag):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    with open("result", 'w') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds) == len(words.split()) == len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write("{} {} {}\n".format(w, t, p))
            fout.write("\n")