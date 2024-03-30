import random
import os
import numpy as np
import time
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)
from dataset_preprocess import MimicDataset
from datasets import Dataset
from transformers import PreTrainedTokenizerFast
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from catboost import CatBoostClassifier
from argparse import ArgumentParser
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def write_json(path, file):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    with open(path, 'w+') as f:
        json.dump(file, f)

def train_corp_iter(dataset): 
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["question"]

def dummy(text):
    return text

def minobjective(weights, truthset, pred1, pred2, pred3):
   weighted_sum = weights[0] * pred1 + weights[1] * pred2 + weights[2] * pred3
   objectiveval = roc_auc_score(truthset, weighted_sum)
   return -objectiveval

def get_args():
    parser = ArgumentParser(description="LLM SELECTOR")
    parser.add_argument("--output_path", type=str)
    arguments = parser.parse_args()
    return arguments

def main(args):
    seed_everything(42)
    dataset_mimic = MimicDataset()
    train_dataset = dataset_mimic.process_data(skip_null=False)
    val_dataset = dataset_mimic.process_test_data()

    LOWERCASE = False
    # VOCAB_SIZE = 30522
    VOCAB_SIZE = 14000000
    # VOCAB_SIZE = 3000
    raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)

    raw_tokenizer.train_from_iterator(train_corp_iter(val_dataset), trainer=trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tokenized_texts_test = []
    for text in tqdm(val_dataset['question']):
        tokenized_texts_test.append(tokenizer.tokenize(text))

    tokenized_texts_train = []
    for text in tqdm(train_dataset['question']):
        tokenized_texts_train.append(tokenizer.tokenize(text))
    
    vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, 
        analyzer = 'word',
        tokenizer = dummy,
        preprocessor = dummy,
        token_pattern = None, strip_accents='unicode')
    vectorizer.fit(tokenized_texts_test)
    vocab = vectorizer.vocabulary_
    print("Vocabulary length", len(vocab))

    vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                                analyzer = 'word',
                                tokenizer = dummy,
                                preprocessor = dummy,
                                token_pattern = None, strip_accents='unicode'
                                )
    tf_train = vectorizer.fit_transform(tokenized_texts_train)
    tf_test = vectorizer.transform(tokenized_texts_test)

    y_train = train_dataset['numerical_label']

    neg_deltas = [0.0, 0.0, 0.0,]
    clf = MultinomialNB(alpha=0.02)
    print("clf fit in progress")
    clf.fit(tf_train, y_train)
    p1 = clf.predict_proba(tf_test)[:, 1]
    tpredictions_mnb = clf.predict_proba(tf_train)[:, 1]
    neg_deltas[0] = 1 - roc_auc_score(y_train, tpredictions_mnb)
    print("clf fit Done!")

    sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber")
    print("sqd fit in progress")
    sgd_model.fit(tf_train, y_train)
    p2 = sgd_model.predict_proba(tf_test)[:, 1]
    tpredictions_sgd = sgd_model.predict_proba(tf_train)[:, 1]
    neg_deltas[1] = 1 - roc_auc_score(y_train, tpredictions_sgd)
    print("SGD Done!")

    cat=CatBoostClassifier(iterations=3400, #task_type='GPU', bootstrap_type='Bernoulli', 
                        verbose=0,
                        l2_leaf_reg=6.5,
                        learning_rate=0.0056,
                        subsample = 0.3,
                        allow_const_label=True,loss_function = 'CrossEntropy')
    try:
        cat.fit(tf_train, y_train)
        predictions_cat = cat.predict_proba(tf_test)[:, 1]
        tpredictions_cat = cat.predict_proba(tf_train)[:, 1]
        neg_deltas[2] = 1 - roc_auc_score(y_train, tpredictions_cat)
        print('done with catboost')
    except Exception:
            print("Skipping catboost in the dev run as it cannot train on test only vocab.")
            
    initial_weights = [1/3, 1/3, 1/3]
    constraints = [{'type': 'eq', 'fun': lambda w: sum(w) - 1}]
    bounds = [(0, 1)] * 3

    print("Attempting to find an optimal set of weights")
    starttime = time.time()
    options = {'maxiter': 2000, 'maxfev': 5000}
    result = minimize(minobjective, initial_weights, args=(y_train, tpredictions_mnb,tpredictions_sgd,tpredictions_cat,),
                        method='Nelder-Mead', bounds=bounds, options=options , constraints=constraints)

    weights = result.x
    print("weights =",weights)
    print("Took",time.time()-starttime,"seconds.")

    final_preds = (weights[0] * p1 + weights[1] * p2 + weights[2] * predictions_cat) / sum(weights)
    threshould = sum(final_preds)/len(final_preds)
    result = (final_preds > threshould).astype(int)

    final_data={}
    for idx , data in enumerate(val_dataset):
        final_data[data["id"]] = result[idx]
    write_json(args.output_path,final_data)
    

if __name__ == "__main__":
    args = get_args()
    main(args)