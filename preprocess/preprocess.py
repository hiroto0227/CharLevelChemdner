import sys, os
sys.path.append("../")
sys.path.append("./")
sys.path.append(os.path.dirname(__file__))
import argparse
from tqdm import tqdm
from sentencepieces import sp_tokenizer
from utils.labels import O, B, I, ZERO
from nltk.tokenize import word_tokenize


def writeSeqsTSV(char_seq, token_seqs, label_seq, outpath, mode="a"):
    def is_end_char_seq(char_seq):
        text = "".join(char_seq)
        tokenized = word_tokenize(text)
        if "." in tokenized:
            return True
        else:
            return False

    num_tokenizer = len(token_seqs)
    end_ixs = []
    ix = 0
    _pre_char_seq = []
    with open(outpath, mode) as f:
        for i in range(len(char_seq)):
            if char_seq[i] == "\n":
                f.write('\n')
            elif char_seq[i] == "." and is_end_char_seq(_pre_char_seq):
                f.write('\n')
                _pre_char_seq = []
            elif char_seq[i] in ["\t", " "]:
                pass
            else:
                row = [char_seq[i]] + [token_seqs[j][i].replace("\n", "") for j in range(num_tokenizer)] + [label_seq[i]]
                f.write("\t".join(row) + "\n")
            _pre_char_seq.append(char_seq[i])
    return True

def fileToCharSeqs(path):
    with open(path + '.txt', 'rt') as f:
        text = f.read()
    with open(path + '.ann', 'rt') as f:
        annotations = f.read().split('\n')
    text_spantokens = textToSpantokens(text, list)
    ann_spantokens = annotationsToSpantokens(annotations)
    char_seq, label_seq = makeTokensAndLabels(text_spantokens, ann_spantokens)
    assert len(char_seq) == len(label_seq), "LengthError! FileToCharSeqs"
    return char_seq, label_seq

def makeTokenSeq(path, tokenizer, mode="Redundant"):
    with open(path + '.txt', 'rt') as f:
        text = f.read()
    token_seqs = []
    for token in tokenizer(text):
        if mode == "Redundant":
            token_seqs.extend([token] * len(token))
        elif mode == "FirstSync":
            token_seqs.extend([token] + ([ZERO] * (len(token) - 1)))
        elif mode == "LastSync":
            token_seqs.extend(([ZERO] * (len(token) - 1)) + [token])
        elif mode == "BiSync":
            token_seqs.extend([token] + ([ZERO] * (len(token) - 2)) + [token])
        else:
            print("--mode choised in Redundant, FirstSync, LastSync, BiSync. Not {}".format(mode))
            exit()
    return token_seqs

def textToSpantokens(text, tokenizer):
    """textをtokenizeし、(token, start_ix, end_ix)のリストとして返す。"""
    spantokens = []
    ix = 0
    end_ix = 1
    for token in tokenizer(text):
        spantokens.append((token, ix, end_ix + len(token) - 1))
        ix += len(token)
        end_ix += len(token)
    return spantokens

def annotationsToSpantokens(annotations):
    spantokens = []
    for annotation in annotations:
        if annotation:
            token = annotation.split('\t')[-1]
            start = int(annotation.split('\t')[1].split(' ')[1])
            end = int(annotation.split('\t')[1].split(' ')[-1])
            if token:
                spantokens.append((token, start, end))
    return sorted(spantokens, key=lambda x: x[1])

def makeTokensAndLabels(text_spantokens, ann_spantokens):
    tokens, labels = [], []
    ann_idx = 0
    pre_label = O
    for token, start, end in text_spantokens:
        if ann_idx >= len(ann_spantokens):
            tokens.append(token)
            labels.append(O)
        elif start == ann_spantokens[ann_idx][1]:
            tokens.append(token)
            labels.append(B)
        elif start > ann_spantokens[ann_idx][1] and end <= ann_spantokens[ann_idx][2]:
            if pre_label == O:
                labels[-1] = B
            tokens.append(token)
            labels.append(I)
        elif start < ann_spantokens[ann_idx][1] or end > ann_spantokens[ann_idx][2]:
            tokens.append(token)
            labels.append(O)
        else:
            print("Labelize Error !!!")
            exit(1)
        if ann_idx < len(ann_spantokens) and end >= ann_spantokens[ann_idx][2]:
            ann_idx += 1
        pre_label = labels[-1]
    return tokens, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser("preprocess for chemdner data")
    parser.add_argument("--input-dir", type=str, help="input directory")
    parser.add_argument("--output-path", type=str, help="output directory")
    parser.add_argument("--sp-models", type=list, nargs='*', help="sentence piece model path")
    parser.add_argument("--mode", type=str, help="choice in Redundant, FirstSync, LastSync")
    opt = parser.parse_args()
    ####### load tokenizers #########
    tokenizers = []
    for sp_model in opt.sp_models:
        sp = sp_tokenizer.SentencePieceTokenizer()  
        sp.load("".join(sp_model))
        tokenizers.append(sp.tokenize)
    fileids = [filename.replace('.txt', '') for filename in os.listdir(opt.input_dir) if filename.endswith('.txt')]
    for i, fileid in tqdm(enumerate(fileids)):
        char_seq, label_seq = fileToCharSeqs(os.path.join(opt.input_dir, fileid))
        token_seqs = [makeTokenSeq(os.path.join(opt.input_dir, fileid), tokenizer, mode=opt.mode) for tokenizer in tokenizers]
        if i == 0:
            writeSeqsTSV(char_seq, token_seqs, label_seq, opt.output_path, mode="w")
        else:
            writeSeqsTSV(char_seq, token_seqs, label_seq, opt.output_path, mode="a")
