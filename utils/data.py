import numpy as np


def load_data(seqfile):
    char_seq, label_seq = [], []
    token_seqs = []
    with open(seqfile, "rt") as f:
        _char, _label = [], []
        for i, line in enumerate(f.read().split("\n")[:-2]):
            splited_line = line.split("\t")
            if i == 0:
                num_token = len(splited_line) - 2
                _tokens = [[] for i in range(num_token)]
                token_seqs = [[] for i in range(num_token)]
            # \nがくるごとにそれぞれの系列を追加し、初期化(\n)が二連続の場合もある。
            if len(splited_line) <= 1:
                if len(_char) >= 1:
                    char_seq.append(_char)
                    label_seq.append(_label)
                    for k, _token in enumerate(_tokens):
                        token_seqs[k].append(_token)
                _char, _label, _tokens = [], [], [[] for i in range(num_token)]
            else:
                _char.append(splited_line[0])
                _label.append(splited_line[-1])
                for j in range(0, num_token):
                    _tokens[j].append(splited_line[j + 1])
    return char_seq, token_seqs, label_seq


def batch_gen(char_seq, tokens_seq, label_seq, batch_size, char_pad_ix=0, label_pad_ix=0, tokens_pad_ix=0, shuffle=True):
    char_batch, label_batch = [], []
    tokens_batch = [[] for i in range(len(tokens_seq))]
    max_len = 0
    # 文の長さでsortすることでpaddingを少なくする。
    # sorted_ix = np.argsort([len(s) for s in char_seq])
    ixs = np.arange(len(char_seq))
    if shuffle:
        np.random.shuffle(ixs)
    for i, ix in enumerate(ixs):
        if max_len < len(char_seq[ix]):
            max_len = len(char_seq[ix])
        char_batch.append(char_seq[ix])
        label_batch.append(label_seq[ix])
        for j, token_seq in enumerate(tokens_seq):
            tokens_batch[j].append(token_seq[ix])
        if (i + 1) % batch_size == 0:
            yield (padding(char_batch, max_len, char_pad_ix),
                   [padding(token_batch, max_len, tokens_pad_ix) for token_batch in tokens_batch],
                   padding(label_batch, max_len, label_pad_ix))
            char_batch, label_batch = [], []
            tokens_batch = [[] for i in range(len(tokens_seq))]
            max_len = 0
            

def padding(batches, max_len, pad_ix):
    pad_batches = []
    for batch in batches:
        pad_length = max_len - len(batch)
        pad_batches.append(list(batch) + [pad_ix for i in range(pad_length)])
    return pad_batches
