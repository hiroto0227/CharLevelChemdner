import os
import 

def make_file_for_sptrain(dirpath, outpath):
   texts = [] 
   fileids = [filename.replace('.txt', '') for filename in os.listdir(dirpath) if filename.endswith('.txt')] 
   for fileid in tqdm(fileids):
       with open(os.path.join(dirpath, fileid + '.txt'), 'rt') as f:
            text = f.read()
       texts.extend(re.split("(\n)", text))
       with open(outpath, "at") as f:
           for text in tqdm(texts):
               if text != "\n":
                   f.writelines([text + '\n'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="sentence piece tokenizer train")
    parser.add_argument('--vocab-size', type=int, default=8000) 
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--train-dir', type=str)
    opt = parser.parse_args()

    for mode in ['train', 'valid']:
        make_file_for_sptrain(os.path.join(opt.train_dir, mode), opt.output_path)
