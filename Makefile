.PHONY: preprocess evaluate

train_define:
	echo "Char Level Redundant + 64000, 32000, 16000, 8000, 4000, 2000\n"

parse_large_corpus:
	#python3.7 large_corpus/parse_PubMed.py --input-dir ./Repository/LargeCorpus/PubMed --output-path ./Repository/LargeCorpus/Corpus.txt

train-sentencepiece:
	#python3.7 sentencepieces/sp_train.py --vocab-size 64000 --input-path ./Repository/LargeCorpus/Corpus.txt;
	#python3.7 sentencepieces/sp_train.py --vocab-size 32000 --input-path ./Repository/LargeCorpus/Corpus.txt;
	#python3.7 sentencepieces/sp_train.py --vocab-size 16000 --input-path ./Repository/LargeCorpus/Corpus.txt;
	#python3.7 sentencepieces/sp_train.py --vocab-size 8000 --input-path ./Repository/LargeCorpus/Corpus.txt;
	#python3.7 sentencepieces/sp_train.py --vocab-size 4000 --input-path ./Repository/LargeCorpus/Corpus.txt;
	#python3.7 sentencepieces/sp_train.py --vocab-size 2000 --input-path ./Repository/LargeCorpus/Corpus.txt;
	mv ./sp*.model ./Repository/SentencePieceModel/;
	mv ./sp*.vocab ./Repository/SentencePieceModel/

pretraining-dataload:
	#python3.7 pretrain/make_pretrain_text.py --input-path ./Repository/LargeCorpus/Corpus.txt --output-path ./Repository/LargeCorpus/pretrain_sp64000.txt --sp-model ./Repository/SentencePieceModel/sp64000.model;
	#python3.7 pretrain/make_pretrain_text.py --input-path ./Repository/LargeCorpus/Corpus.txt --output-path ./Repository/LargeCorpus/pretrain_sp32000.txt --sp-model ./Repository/SentencePieceModel/sp32000.model;
	#python3.7 pretrain/make_pretrain_text.py --input-path ./Repository/LargeCorpus/Corpus.txt --output-path ./Repository/LargeCorpus/pretrain_sp16000.txt --sp-model ./Repository/SentencePieceModel/sp16000.model;
	#python3.7 pretrain/make_pretrain_text.py --input-path ./Repository/LargeCorpus/Corpus.txt --output-path ./Repository/LargeCorpus/pretrain_sp8000.txt --sp-model ./Repository/SentencePieceModel/sp8000.model;
	#python3.7 pretrain/make_pretrain_text.py --input-path ./Repository/LargeCorpus/Corpus.txt --output-path ./Repository/LargeCorpus/pretrain_sp4000.txt --sp-model ./Repository/SentencePieceModel/sp4000.model;
	#python3.7 pretrain/make_pretrain_text.py --input-path ./Repository/LargeCorpus/Corpus.txt --output-path ./Repository/LargeCorpus/pretrain_sp2000.txt --sp-model ./Repository/SentencePieceModel/sp2000.model;

pretraining-fasttext:
	~/fastText-0.1.0/fasttext skipgram -input ./Repository/LargeCorpus/pretrain_sp64000.txt -output ./Repository/Pretrained/ft_pretrain_sp64000.model -dim 50  -epoch 20;
	~/fastText-0.1.0/fasttext skipgram -input ./Repository/LargeCorpus/pretrain_sp32000.txt -output ./Repository/Pretrained/ft_pretrain_sp32000.model -dim 50  -epoch 20;
	~/fastText-0.1.0/fasttext skipgram -input ./Repository/LargeCorpus/pretrain_sp16000.txt -output ./Repository/Pretrained/ft_pretrain_sp16000.model -dim 50  -epoch 20;
	~/fastText-0.1.0/fasttext skipgram -input ./Repository/LargeCorpus/pretrain_sp8000.txt -output ./Repository/Pretrained/ft_pretrain_sp8000.model -dim 50  -epoch 20;
	~/fastText-0.1.0/fasttext skipgram -input ./Repository/LargeCorpus/pretrain_sp4000.txt -output ./Repository/Pretrained/ft_pretrain_sp4000.model -dim 50  -epoch 20;
	~/fastText-0.1.0/fasttext skipgram -input ./Repository/LargeCorpus/pretrain_sp2000.txt -output ./Repository/Pretrained/ft_pretrain_sp2000.model -dim 50  -epoch 20;

pretraining-glove:
	python3.7 pretrain/glove.py --vocab-min-count 5 --input-path ./Repository/LargeCorpus/pretrain_sp64000.txt --output-path ./Repository/Pretrained/gv_pretrain_sp64000.model --vector-size 50
	python3.7 pretrain/glove.py --vocab-min-count 5 --input-path ./Repository/LargeCorpus/pretrain_sp32000.txt --output-path ./Repository/Pretrained/gv_pretrain_sp32000.model --vector-size 50
	python3.7 pretrain/glove.py --vocab-min-count 5 --input-path ./Repository/LargeCorpus/pretrain_sp16000.txt --output-path ./Repository/Pretrained/gv_pretrain_sp16000.model --vector-size 50
	python3.7 pretrain/glove.py --vocab-min-count 5 --input-path ./Repository/LargeCorpus/pretrain_sp8000.txt --output-path ./Repository/Pretrained/gv_pretrain_sp8000.model --vector-size 50
	python3.7 pretrain/glove.py --vocab-min-count 5 --input-path ./Repository/LargeCorpus/pretrain_sp4000.txt --output-path ./Repository/Pretrained/gv_pretrain_sp4000.model --vector-size 50
	python3.7 pretrain/glove.py --vocab-min-count 5 --input-path ./Repository/LargeCorpus/pretrain_sp2000.txt --output-path ./Repository/Pretrained/gv_pretrain_sp2000.model --vector-size 50

preprocess:
	python3.7 preprocess/preprocess.py --input-dir ./Repository/Chemdner/train --output-path ./Repository/SeqData/train_bl.csv --sp-model;
	python3.7 preprocess/preprocess.py --input-dir ./Repository/Chemdner/valid --output-path ./Repository/SeqData/valid_bl.csv --sp-model;
	python3.7 preprocess/preprocess.py --input-dir ./Repository/Chemdner/test --output-path ./Repository/SeqData/test_bl.csv --sp-model;

train:
	python3.7 model/charRedundant.py --mode train --config-path model/baseline.config;

predict:
	python3.7 model/charRedundant.py --mode predict --config-path model/baseline.config

evaluate:
	python3.7 evaluate/evaluate.py --input-path ./Repository/SeqData/predicted_bl.csv;
