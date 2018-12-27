import sentencepiece as spm
import re


class SentencePieceTokenizer():
    def __init__(self):
        self.spe = spm.SentencePieceProcessor()

    def train(self, text_path, vocab_size):
        spm.SentencePieceTrainer.Train('--normalization_rule_name=identity --input={} --model_prefix=sp{} --vocab_size={} --model_type=unigram'.format(text_path, vocab_size, vocab_size))

    def load(self, sp_model_path):
        self.spe.Load(sp_model_path)

    def tokenize(self, text):
        tokens = []
        pre_token = ""
        for i, token in enumerate(self.spe.EncodeAsPieces(text)):
            if token != '':
                if i > 0 and token.startswith('▁'):
                    if pre_token == '\n':
                        tokens.append('\n')
                    else:
                        tokens.append(' ')
                token = token.replace('▁', '')
                tokens.extend(list(filter(None, re.split('(,|\(|\)|-)', token))))
            pre_token = token
        return tokens
