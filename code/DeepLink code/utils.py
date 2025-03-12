from __future__ import absolute_import, division, print_function

import argparse
import os

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from imblearn.under_sampling import RandomUnderSampler
from torch.utils.data import Dataset

from preprocessor import textProcess

# Default Parameters
epoch = 100
eval_batch_size = 16
train_batch_size = 32
seed_num = 42
do_advTrain = True
flag_train = True
flag_test = True

key = '#'
pro = 'F'

mode = "DBACE"
best_f = -1.0
inline_dataset_best_f = -1.0
test_dataset = "B"

# Important Dir
result_path = 'results'  # your result path
data_dir = r'../dataset'  # your path to dataset dir

# model path
model_output_path = r'../models/DeepLink_saved_models'  # your saved model path
text_model_path = r"../models/robert-large"
code_model_path = r"../models/codebert-base"


def set_seed(seed=42):
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 text,
                 label,
                 issue_key,
                 commit_sha,
                 pro=-1.0):
        self.text = text
        self.label = label
        self.issue_key = issue_key,
        self.commit_sha = commit_sha,
        self.pro = pro


def MySubSampler(df, args):
    X, y = df[['Issue_KEY', 'Commit_SHA', 'Issue_Text', 'Commit_Text', 'Commit_Code']], df['label']
    # 下采样
    # try:
    #     if args.key in ['AIRFLOW', 'AMBARI', 'ARROW', 'CALCITE', 'CASSANDRA', 'FLINK', 'FREEMARKER', 'GROOVY', 'IGNITE', 'NETBEANS']:
    #         X_resampled, y_resampled = X, y
    #     else:
    #         rus = RandomUnderSampler(random_state=args.seed, sampling_strategy=1/2)
    #         X_resampled, y_resampled = rus.fit_resample(X, y)
    # except:
    #     X_resampled, y_resampled = X, y
    rus = RandomUnderSampler(random_state=args.seed)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    df = pd.concat([X_resampled, y_resampled], axis=1)
    return df.sample(frac=1)


def getargs():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--mode", default=mode, type=str, help="mode")
    parser.add_argument("--best_f", default=best_f, type=float, help="best_f")
    parser.add_argument("--inline_dataset_best_f", default=inline_dataset_best_f, type=float,
                        help="inline_dataset_best_f")
    parser.add_argument("--test_dataset", default=test_dataset, type=str, help="test_dataset")

    parser.add_argument("--data_dir", default=data_dir, type=str, help="data_dir")
    parser.add_argument("--output_dir", default=model_output_path, type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--result_dir", default=result_path, type=str,
                        help="The output directory where the result files will be written.")

    # Other parameters
    parser.add_argument("--text_model_path", default=text_model_path, type=str,
                        help="The NL-NL model checkpoint for weights initialization.")
    parser.add_argument("--code_model_path", default=code_model_path, type=str,
                        help="The NL-PL model checkpoint for weights initialization.")
    parser.add_argument("--tokenizer_name", default=text_model_path, type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", default=flag_train, type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_test", default=flag_test, type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=train_batch_size, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=eval_batch_size, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=seed_num,
                        help="random seed for initialization")
    parser.add_argument('--num_train_epochs', type=int, default=epoch,
                        help="num_train_epochs")
    parser.add_argument("--key", default=key, type=str,
                        help="Key of the project.")
    parser.add_argument("--pro", default=pro, type=str,
                        help="The used project.")
    parser.add_argument("--num_class", default=2, type=int,
                        help="The number of classes.")
    parser.add_argument("--do_advTrain", default=do_advTrain, type=bool,
                        help="Whether to conduct adversarial training.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.n_gpu = 1
    args.device = device
    return args


class TextDataset(Dataset):
    def __init__(self, args, file_path=None):
        self.Issue_text_examples = []
        self.Commit_text_examples = []
        self.code_examples = []
        if 'TRAIN' in file_path:
            df_link = MySubSampler(pd.read_csv(file_path), args)
        else:
            df_link = pd.read_csv(file_path)
        # 创建嵌入
        # self.embedding_model = word2vec.KeyedVectors.load_word2vec_format(
        #     model_output_path + '/word2vec-google-news-300/GoogleNews-vectors-negative300.bin',
        #     binary=True)  # 加载Google词表
        # self.issue_embedding_model = Word2Vec(vector_size=50, window=5, min_count=1, sg=1)
        # self.commit_embedding_model = Word2Vec(vector_size=50, window=5, min_count=1, sg=1)
        # self.code_embedding_model = Word2Vec(vector_size=50, window=5, min_count=1, sg=1)

        issue_texts = df_link['Issue_Text'].apply(lambda x: textProcess(x, args.key))
        commit_texts = df_link['Commit_Text'].apply(lambda x: textProcess(x, args.key))
        commit_codes = df_link['Commit_Code'].apply(lambda x: textProcess(x, args.key))

        if 'TRAIN' in file_path:
            self.issue_embedding_model = self.train_skipgram(issue_texts)
            self.commit_embedding_model = self.train_skipgram(commit_texts)
            self.code_embedding_model = self.train_skipgram(commit_codes)

            self.issue_embedding_model.save("issue_word2vec.model")
            self.commit_embedding_model.save("commit_word2vec.model")
            self.code_embedding_model.save("code_word2vec.model")
        else:
            self.issue_embedding_model = Word2Vec.load("issue_word2vec.model")
            self.commit_embedding_model = Word2Vec.load("commit_word2vec.model")
            self.code_embedding_model = Word2Vec.load("code_word2vec.model")

        df_link['Issue_Embedding'] = issue_texts.apply(lambda x: self.get_embeddings(x, self.issue_embedding_model))
        df_link['Commit_Text_Embedding'] = commit_texts.apply(lambda x: self.get_embeddings(x, self.commit_embedding_model))
        df_link['Commit_Code_Embedding'] = commit_codes.apply(lambda x: self.get_embeddings(x, self.code_embedding_model))

        for i_row, row in df_link.iterrows():
            self.Issue_text_examples.append(
                InputFeatures(row['Issue_Embedding'], row['label'], row['Issue_KEY'], row['Commit_SHA']))
            self.Commit_text_examples.append(
                InputFeatures(row['Commit_Text_Embedding'], row['label'], row['Issue_KEY'], row['Commit_SHA']))
            self.code_examples.append(
                InputFeatures(row['Commit_Code_Embedding'], row['label'], row['Issue_KEY'], row['Commit_SHA']))
        assert len(self.Issue_text_examples) == len(self.code_examples), 'ErrorLength'

    def train_skipgram(self, sentences, vector_size=50, window=5, min_count=1):
        # 训练Skip-Gram模型
        model = Word2Vec(sentences.to_list(), vector_size=vector_size, window=window, min_count=min_count)
        return model

    # 将文本转化为嵌入
    def get_embeddings(self, text, embedding_model):
        embeddings = []
        for word in text:
            try:
                embeddings.append(embedding_model.wv[word])
            except:
                continue
        if len(embeddings) == 0:
            embeddings.append(np.zeros(50, dtype=np.float32))
        mean_embedding = np.mean(embeddings, axis=0)
        return mean_embedding

    def __len__(self):
        return len(self.Issue_text_examples)

    def __getitem__(self, i):
        return (torch.tensor(self.Issue_text_examples[i].text),
                torch.tensor(self.Commit_text_examples[i].text),
                torch.tensor(self.code_examples[i].text),
                torch.tensor(self.code_examples[i].label),
                self.Issue_text_examples[i].issue_key,
                self.Issue_text_examples[i].commit_sha)
