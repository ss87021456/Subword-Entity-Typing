import pandas as pd
import numpy as np
from scipy import sparse
import argparse, json
import pickle as pkl
from utils import split_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support 
from fastText_model import fastText # pretrain-model
from keras.preprocessing import text, sequence
from keras import backend as K
from keras.layers import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from nn_model import BLSTM, CNN
from tqdm import tqdm

# simply output prediction
# CUDA_VISIBLE_DEVICES=1 python ./src/evaluation.py --model_path=... --model_type=[BLSTM,CNN] [--subword] [--attention]

# visualize
# CUDA_VISIBLE_DEVICES=1 python ./src/evaluation.py --model_path=... --model_type=[BLSTM,CNN] [--subword] [--attention] --visualize > visualize.txt

# Feature-parameter
MAX_NUM_WORDS = 30000
MAX_NUM_MENTION_WORDS = 20000
MAX_SEQUENCE_LENGTH = 40
MAX_MENTION_LENGTH = 5 # 15 if subowrd else 5
EMBEDDING_DIM = 100

# Hyper-parameter
batch_size = 64
epochs = 5

def p_r_f(pred, label):
    overlap_count = len(set(pred) & set(label))
    if len(pred) != 0:
        precision = overlap_count / len(pred)
    else:
        precision = 0

    recall = overlap_count / len(label)
    if (precision + recall) != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.

    return precision, recall, f1

# Set memory constraint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

# Feature-parameter
MAX_NUM_WORDS = 30000
MAX_NUM_MENTION_WORDS = 20000
MAX_SEQUENCE_LENGTH = 40
MAX_MENTION_LENGTH = 5 # 15 if subowrd else 5
EMBEDDING_DIM = 100

# Hyper-parameter
batch_size = 64
epochs = 5

# Set memory constraint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

def run(model_dir, model_type, model_path, subword=False, attention=False, visualize=False):
    print(model_dir, model_type, model_path)
    # Parse directory name
    if not model_dir.endswith("/"):
        model_dir += "/"
    # Load models
    if subword:
        mlb = pkl.load(open(model_dir + "mlb_w_subword_filter.pkl", 'rb'))
    else:
        mlb = pkl.load(open(model_dir + "mlb_wo_subword_filter.pkl", 'rb'))
    
    label_num = len(mlb.classes_)
    
    # Building Model
    print("Building computational graph...")
    if model_type == "BLSTM":
        print("Building default BLSTM mode with attention:",attention,"subword:",subword)
        model = BLSTM(label_num=label_num, sentence_emb=None, mention_emb=None, attention=attention, subword=subword, mode='concatenate', dropout=0.1)
    elif model_type == "CNN":
        print("Building default CNN mode with attention:",attention,"subword:",subword)
        model = CNN(label_num=label_num, sentence_emb=None, mention_emb=None, attention=attention, subword=subword, mode='concatenate', dropout=0.1)

    print(model.summary())

    file_path =  model_type + "-weights-{epoch:02d}.hdf5"
    model_name = model_type + "-weights-00.hdf5"
    if attention:
        file_path = "Attention-" + file_path
        model_name = "Attention-" + model_name
    if subword:
        file_path = "Subword" + file_path
        model_name = "Subword" + model_name

    checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=False, mode='min') # Save every epoch
    early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    callbacks_list = [checkpoint, early] #early

    print("Loading testing data...")
    if subword:
        X_test = pkl.load(open(model_dir + "testing_data_w_subword_filter.pkl", 'rb'))
        X_test_mention = pkl.load(open(model_dir + "testing_mention_w_subword_filter.pkl", 'rb'))
        y_test = pkl.load(open(model_dir + "testing_label_w_subword_filter.pkl", 'rb'))
    else:
        X_test = pkl.load(open(model_dir + "testing_data_wo_subword_filter.pkl", 'rb'))
        X_test_mention = pkl.load(open(model_dir + "testing_mention_wo_subword_filter.pkl", 'rb'))
        y_test = pkl.load(open(model_dir + "training_label_wo_subword_filter.pkl", 'rb'))

    print("loading file..")
    if subword:
        dataset = pd.read_csv("./data/smaller_preprocessed_sentence_keywords_labeled_subwords.tsv", sep='\t', names=['label','context','mention'])
    else:
        dataset = pd.read_csv("./data/smaller_preprocessed_sentence_keywords_labeled.tsv", sep='\t', names=['label','context','mention'])

    X = dataset['context'].values
    y = dataset['label'].values
    mentions = dataset['mention'].values
    train_index = pkl.load(open(model_dir + "train_index.pkl", 'rb'))
    test_index = pkl.load(open(model_dir + "test_index.pkl", 'rb'))
    X_train_text, X_test_text = X[train_index], X[test_index]
    X_train_mention_text, X_test_mention_text = mentions[train_index], mentions[test_index]
    del X, mentions
    
    # Visualize number of rows
    test_size = 1000 if visualize else None

    trained_weight_file = model_path
    model.load_weights(trained_weight_file)
    y_pred = model.predict([X_test[:test_size], X_test_mention[:test_size]])
    y_pred[y_pred >= 0.5] = 1.
    y_pred[y_pred < 0.5] = 0.
    y_pred = sparse.csr_matrix(y_pred)

    print("inverse_transform result...")
    y_pred = mlb.inverse_transform(y_pred)
    y_test_ = mlb.inverse_transform(y_test[:test_size])

    label_dict = json.load(open('data/label.json', "r"))
    inv_label_dict = {v: k for k, v in label_dict.items()}
    
    filename = trained_weight_file[:-5]+"_result.txt"

    print("Start output result...")
    if not visualize:
        with open(filename, "w") as f:
            for prediction, lab in tqdm(zip(y_pred, y_test_)):
                pred = list(prediction)
                pred.sort()
                label = list(lab)
                label.sort()
                pred = list(map(str, pred))
                pred = ",".join(pred)
                label = list(map(str, label))
                label = ",".join(label)
                f.write(pred + '\t')
                f.write(label + '\n')
                print("Writing file to", filename)
    else:
        for i in range(test_size):
            pred = list(y_pred[i])
            pred.sort()
            label = list(y_test_[i])
            label.sort()
            precision, recall, f1 = p_r_f(pred, label)
            pred = [inv_label_dict[j] for j in pred]
            label = [inv_label_dict[j] for j in label]
            temp = []
            temp.append(X_train_text[i])
            temp_ = []
            temp_.append(X_train_mention_text[i])
            if 'toxin' in temp_[0]:
                print("Sentence: ",temp)
                print("Mention",temp_)
                print("Prediction",pred)
                print("Label",label)
                score = "Precision: {:3.3f} Recall {:3.3f} F1 {:3.3f}".format(precision, recall, f1)
                print(score + "\n")

    K.clear_session()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subword", action="store_true" , help="Use subword or not")
    parser.add_argument("--attention",action="store_true", help="Use attention or not")
    parser.add_argument("--visualize",action="store_true", help="Visualize or not")
    parser.add_argument("--model_dir", nargs='?', type=str, default="model/", 
                        help="Directory to load models. [Default: \"model/\"]")
    parser.add_argument("--model_type", nargs='?', type=str, default="BLSTM",
                        help="different model architecture BLTSM or CNN [Default: \"BLSTM/\"]")
    parser.add_argument("--model_path", nargs='?', type=str, default="None", help="path to weighted")
    args = parser.parse_args()

    run(args.model_dir, args.model_type, args.model_path, args.subword, args.attention, args.visualize)
