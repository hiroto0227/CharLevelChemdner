import sys, os
import backtrace
import argparse
import time
import numpy as np
from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
from torchcrf import CRF

sys.path.append("../")
sys.path.append("./")
from utils import data, vocab, trainutils, labels
from utils.trainutils import get_variable
from load_config import load_state
from pretrain.pretrain import load_word2vec, make_pretrain_embed


def writeSeqsTSV(char_seq, token_seqs, label_seq, pred_seq, outpath, mode="a"):
    num_tokenizer = len(token_seqs)
    with open(outpath, mode) as f:
        for i in range(len(char_seq)):
            pred_label = labels.O if pred_seq[i] in [labels.UNK, labels.PAD] else pred_seq[i]
            row = [char_seq[i]] + [token_seqs[j][i].replace("\n", "") for j in range(num_tokenizer)] + [label_seq[i]] + [pred_label]
            f.write("\t".join(row) + "\n")
        f.write("\n")
    return True


class BiLSTMLSTMCRF(nn.Module):

    def __init__(self, char_vocab_dim, token_vocab_dims, label_dim, token_embed_dim, char_embed_dim, lstm_dim, dropout, pretrain_embeds, training, unk_ix=None, zero_ixs=[], gpu=False):
        super().__init__()
        self.training = training
        self.gpu = gpu
        self.lstm_dim = lstm_dim
        self.num_token_layer = len(token_vocab_dims)
        self.unk_ix = unk_ix
        self.char_vocab_dim = char_vocab_dim
        self.token_vocab_dims = token_vocab_dims
        unk_embed_scale = 3
        
        # init char embedding
        self.char_embedding = nn.Embedding(char_vocab_dim, char_embed_dim)
        if unk_ix:
            self.char_embedding.weight.data[unk_ix] = torch.from_numpy(np.random.randn(char_embed_dim) / unk_embed_scale)
        for zero_ix in zero_ixs:
            self.char_embedding.weight.data[zero_ix] = torch.from_numpy(np.zeros(char_embed_dim))
        
        # load pretrain
        self.token_embeddings = [nn.Embedding(token_vocab_dim, token_embed_dim) for token_vocab_dim in token_vocab_dims]
        for i, pretrain_embed in enumerate(pretrain_embeds):
            if unk_ix:
                pretrain_embed[unk_ix] = np.random.randn(token_embed_dim) / unk_embed_scale
            for zero_ix in zero_ixs:
                pretrain_embed[zero_ix] = np.zeros(token_embed_dim)
            self.token_embeddings[i].weight.data.copy_(torch.from_numpy(pretrain_embed))
          
        
        self.lstm = nn.LSTM(char_embed_dim + token_embed_dim * self.num_token_layer, lstm_dim, bidirectional=True)
        self.droplstm = nn.Dropout(dropout)
        self.tanh = nn.Linear(lstm_dim * 2, label_dim)
        self.crf = CRF(label_dim)
        
        if gpu > 0:
            self.char_embedding.cuda()
            self.droplstm.cuda()
            [self.token_embeddings[i].cuda() for i in range(len(self.token_embeddings))]
            self.lstm.cuda()
            self.tanh.cuda()
            self.crf.cuda()
            print("=========== use GPU =============")

        print("token_vocab_dims: {}".format(self.token_vocab_dims))

    def init_hidden(self, batch_size):
        return (get_variable(torch.zeros(2, batch_size, self.lstm_dim), gpu=self.gpu),
                get_variable(torch.zeros(2, batch_size, self.lstm_dim), gpu=self.gpu))

    def unk2vec(self, token_ids, token_embed, token2vec, token_seq, unk_id=1):
        seq_length, batch_size, _ = token_embed.shape
        for batch_i in range(batch_size):
            for seq_i in range(seq_length): 
                if token_ids[seq_i][batch_i] == unk_id:
                    try:
                        token_embed[seq_i][batch_i] = torch.from_numpy(token2vec[token_seq[batch_i][seq_i]])
                    except KeyError:
                        try:
                            token_embed[seq_i][batch_i] = torch.from_numpy(token2vec[token_seq[batch_i][seq_i].lower()])
                        except KeyError:
                            pass
        return token_embed

    def _forward(self, char, tokens, tokens_seq=None, token2vecs=None):
        batch_size = char.shape[1]
        self.lstm_hidden = self.init_hidden(batch_size)
        char_embed = self.char_embedding(char)
        token_embeds = []
        for i, token in enumerate(tokens):
            if self.token_vocab_dims[i] <= max(torch.max(token, dim=1)[0]):
                print("======= Out of Index =======")
                token[token >= self.token_vocab_dims[i]] = self.unk_ix
            token_embed = self.token_embeddings[i](token)
            if not self.training and tokens_seq and token2vecs:
                token_embed = self.unk2vec(token, token_embed, token2vecs[i], tokens_seq[i])
            token_embeds.append(token_embed)
        cat_embed = get_variable(torch.cat(([char_embed] + token_embeds), dim=2), gpu=self.gpu)
        lstm_out, self.lstm_hidden = self.lstm(cat_embed, self.lstm_hidden)  # (seq_length, bs, word_hidden_dim)
        lstm_out = self.droplstm(lstm_out)
        out = self.tanh(lstm_out)  # (seq_length, bs, tag_dim)
        return out

    def loss(self, char, tokens, label):
        out = self._forward(char, tokens)
        #log_likelihood = self.crf.neg_log_likelihood_loss(out, label)
        #return log_likelihood
        # log_likelihoodを最大にすれば良いが、最小化するので-1をかけている。
        log_likelihood = self.crf(out, label)
        return -1 * log_likelihood

    def forward(self, word, char, tokens_seq=None, token2vecs=None):
        out = self._forward(word, char, tokens_seq=tokens_seq, token2vecs=token2vecs)
        #decoded = self.crf.decode(out)
        decoded = torch.FloatTensor(self.crf.decode(out))
        return decoded


class CharRedundant:
    def __init__(self, args):
        print("Char Redundant")
        self.args = args

    def train(self):
        # char_seq, label_seq <- (seq_length, char_length)
        # tokens_seq <- (num_tokenizer, seq_length, token_length)
        char_seq, token_seqs, label_seq = data.load_data(self.args.train_path)
        char_encoder = vocab.LabelEncoder()
        char_vec = char_encoder.fit_transform(char_seq)
        label_encoder = vocab.LabelEncoder()
        label_vec = label_encoder.fit_transform(label_seq)
        token_vecs, token_encoders = [], []
        for token_seq in token_seqs:
            token_encoder = vocab.LabelEncoder()
            token_vec = token_encoder.fit_transform(token_seq)
            token_vecs.append(token_vec)
            token_encoders.append(token_encoder)
        # load pretrain embedding
        pretrain_embeds = []
        if self.args.word2vec_path:
            for word2vec_path, token_encoder in zip(self.args.word2vec_path, token_encoders):
                print("===== {}, vocab size={} =====".format(word2vec_path, len(token_encoder.label2id)))
                word2vec = load_word2vec(word2vec_path)
                pretrain_embed = make_pretrain_embed(word2vec, token_encoder.label2id, self.args.token_embed_dim)
                pretrain_embeds.append(pretrain_embed)
        
        model = BiLSTMLSTMCRF(char_vocab_dim=len(char_encoder.label2id), 
                              token_vocab_dims=[len(te.label2id) for te in token_encoders],
                              label_dim=len(label_encoder.label2id),
                              token_embed_dim=self.args.token_embed_dim,
                              char_embed_dim=self.args.char_embed_dim,
                              lstm_dim=self.args.lstm_dim, 
                              pretrain_embeds=pretrain_embeds,
                              dropout=self.args.dropout,
                              training=True,
                              unk_ix=1,
                              zero_ixs=[0, 2],
                              gpu=self.args.gpu)
        print(model)
        optimizer = optim.SGD(model.parameters(), lr=self.args.initial_rate, weight_decay=self.args.weight_decay)
        print("seq length: {}".format(len(char_vec)))
        for epoch in range(1, self.args.epoch + 1):
            print("\n\n================ {} epoch ==================".format(epoch))
            start = time.time()
            loss_per_epoch = 0
            for i, (char_batch, tokens_batch, label_batch) in tqdm(enumerate(data.batch_gen(char_vec, token_vecs, label_vec, self.args.batchsize))):
                try:
                    model.zero_grad()
                    model.train()
                    char_batch = trainutils.get_variable(torch.LongTensor(char_batch), gpu=self.args.gpu).transpose(1, 0)
                    tokens_batch = [trainutils.get_variable(torch.LongTensor(token_batch), gpu=self.args.gpu).transpose(1, 0) for token_batch in tokens_batch]
                    label_batch = trainutils.get_variable(torch.LongTensor(label_batch), gpu=self.args.gpu).transpose(1, 0)
                    #print(char_batch.shape)
                    loss = model.loss(char_batch, tokens_batch, label_batch) / char_batch.shape[1]
                    #print("loss: {}".format(loss))
                    loss.backward()
                    optimizer.step()
                    loss_per_epoch += float(loss)
                except RuntimeError:
                    tpe, v, tb = sys.exc_info()
                    backtrace.hook(reverse=True, strip_path=True, tb=tb, tpe=tpe, value=v)
            print("loss_per_epoch: {}\ntime_per_epoch: {}".format(loss_per_epoch, time.time() - start))
            if epoch % 10 == 0:
                torch.save(model.state_dict(), self.args.save_model + str(epoch))
                print("model saved! {}".format(self.args.save_model + str(epoch)))
        torch.save(model.state_dict(), self.args.save_model)

    def predict(self):
        if os.path.exists(self.args.predicted_path):
            os.remove(self.args.predicted_path)
        
        char_seq, token_seqs, label_seq = data.load_data(self.args.train_path)
        char_encoder = vocab.LabelEncoder()
        char_encoder.fit(char_seq)
        label_encoder = vocab.LabelEncoder()
        label_encoder.fit(label_seq)
        token_encoders = []
        for token_seq in token_seqs:
            token_encoder = vocab.LabelEncoder()
            token_encoder.fit(token_seq)
            token_encoders.append(token_encoder)
        
        # load pretrain embedding
        pretrain_embeds = []
        token2vecs = []
        if self.args.word2vec_path:
            for word2vec_path, token_encoder in zip(self.args.word2vec_path, token_encoders):
                print("===== {}, vocab size={} =====".format(word2vec_path, len(token_encoder.label2id)))
                token2vec = load_word2vec(word2vec_path)
                pretrain_embeds.append(make_pretrain_embed(token2vec, token_encoder.label2id, self.args.token_embed_dim))
                token2vecs.append(token2vec)

        model = BiLSTMLSTMCRF(char_vocab_dim=len(char_encoder.label2id), 
                              token_vocab_dims=[len(te.label2id) for te in token_encoders],
                              label_dim=len(label_encoder.label2id),
                              token_embed_dim=self.args.token_embed_dim,
                              char_embed_dim=self.args.char_embed_dim,
                              lstm_dim=self.args.lstm_dim,
                              pretrain_embeds=pretrain_embeds,
                              dropout=self.args.dropout,
                              training=False,
                              unk_ix=1,
                              zero_ixs=[0, 2],
                              gpu=False)
        model.load_state_dict(torch.load(self.args.load_model))
        print(model)

        char_seq, token_seqs, label_seq = data.load_data(self.args.test_path)
        for i in tqdm(range(len(char_seq))):
            model.eval()
            char_vec = char_encoder.transform([char_seq[i]])
            label_vec = label_encoder.transform([label_seq[i]])
            token_vecs = []
            for j, token_seq in enumerate(token_seqs):
                token_vecs.append(token_encoders[j].transform([token_seq[i]]))
            
            char_batch = torch.LongTensor(char_vec).transpose(1, 0)
            tokens_batch = [torch.LongTensor(token_vec).transpose(1, 0) for token_vec in token_vecs]
            print(char_batch.shape)
            tokens_seq = [[token_seqs[j][i] for k in range(1)] for j in range(len(token_seqs))]  # batch_size=1
            pred_label_ids = model(char_batch, tokens_batch, tokens_seq, token2vecs)
            pred_seq = label_encoder.inverse_transform(pred_label_ids)[0]
            
            _token_seqs = [token_seq[i] for token_seq in token_seqs]
            writeSeqsTSV(char_seq[i], _token_seqs, label_seq[i], pred_seq, outpath=self.args.predicted_path, mode="a")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("char redundant training")
    parser.add_argument("--mode", type=str, choices=["train", "predict"], help="config data path.")
    parser.add_argument("--config-path", type=str, help="config data path.")
    parser.add_argument("--word2vec-path", type=list, nargs='+', help="config data path.")
    opt = parser.parse_args()
    args = load_state(opt.config_path, opt)
    if args.word2vec_path:
        args.word2vec_path = ["".join(path) for path in args.word2vec_path]

    char_redundant = CharRedundant(args)
    if opt.mode == "train":
        char_redundant.train()
    elif opt.mode == "predict":
        char_redundant.predict()
