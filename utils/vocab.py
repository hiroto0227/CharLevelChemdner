from utils import labels


class LabelEncoder():
    def __init__(self):
        # PADDINGは0, UNKNOWNは1, ZEROは2とする。
        self.label2id = {labels.PAD: 0, labels.UNK: 1, labels.ZERO: 2}
        self.id2label = [labels.PAD, labels.UNK, labels.ZERO]

    def fit(self, seq):
        all_set = set([l for line in seq for l in line])
        self.id2label.extend(sorted(list(all_set)))
        self.label2id.update({v: i for i, v in enumerate(self.id2label)})

    def transform(self, seq):
        vec = []
        for line in seq:
            _vec = []
            for label in line:
                try:
                    _vec.append(self.label2id[label])
                except KeyError:
                    _vec.append(self.label2id[labels.UNK])
            vec.append(_vec)
        return vec

    def fit_transform(self, seq):
        self.fit(seq)
        vec = self.transform(seq)
        return vec

    def inverse_transform(self, vec):
        seq = []
        for line in vec:
            _seq = []
            for _id in line:
                _seq.append(self.id2label[int(_id)])
            seq.append(_seq)
        return seq
