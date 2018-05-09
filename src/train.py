import pandas as pd
import numpy as np
from scipy import sparse
import argparse
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

# Training w/o pretrained
# CUDA_VISIBLE_DEVICES=0 python ./src/train.py --mode=[CNN,BLSTM]
# Training w/ pretrained
# CUDA_VISIBLE_DEVICES=0 python ./src/train.py --pre --emb=data/FastText_embedding.vec --mode=[CNN,BLSTM]

# Additional option --subword --attention


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

def run(model_dir, model_type, pre=False, embedding=None, subword=False, attention=False):
    # Parse directory name
    if not model_dir.endswith("/"):
        model_dir += "/"
    ## Load models
    if subword:
        mlb = pkl.load(open(model_dir + "mlb_w_subword_filter.pkl", 'rb'))
        tokenizer = pkl.load(open(model_dir + "tokenizer_w_subword_filter.pkl", 'rb'))
    else:
        mlb = pkl.load(open(model_dir + "mlb_wo_subword_filter.pkl", 'rb'))
        tokenizer = pkl.load(open(model_dir + "tokenizer_wo_subword_filter.pkl", 'rb'))
    
    word_index = tokenizer.word_index
    label_num = len(mlb.classes_)

    if pre:
        print("Loading pre-trained embedding model...")
        embeddings_index = fastText(embedding)

        print("Preparing embedding matrix...")
        # prepare embedding matrix
        num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
        embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
        for word, i in word_index.items():
            if i >= MAX_NUM_WORDS:
                continue
            embedding_vector = embeddings_index[word]
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
        embedding_layer = Embedding(num_words,EMBEDDING_DIM,weights=[embedding_matrix],input_length=MAX_SEQUENCE_LENGTH,trainable=True)
    
    # Building Model
    print("Building computational graph...")
    if model_type == "BLSTM":
        print("Building default BLSTM mode with attention:",attention,"subword:",subword)
        if pre:
            model = BLSTM(label_num=label_num, sentence_emb=embedding_layer, mention_emb=None, attention=attention, subword=subword, mode='concatenate', dropout=0.1)
        else:
            model = BLSTM(label_num=label_num, sentence_emb=None, mention_emb=None, attention=attention, subword=subword, mode='concatenate', dropout=0.1)
    elif model_type == "CNN":
        print("Building default CNN mode with attention:",attention,"subword:",subword)
        if pre:
            model = CNN(label_num=label_num, sentence_emb=embedding_layer, mention_emb=None, attention=attention, subword=subword, mode='concatenate', dropout=0.1)
        else:
            model = CNN(label_num=label_num, sentence_emb=None, mention_emb=None, attention=attention, subword=subword, mode='concatenate', dropout=0.1)

    print(model.summary())
    #exit()

    file_path =  model_type + "-weights-{epoch:02d}.hdf5"   # for keras to save model each epoch
    model_name = model_type + "-weights-00.hdf5"            # deal with model_name   
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
        print(X_test_mention.shape)
        exit()
        y_test = pkl.load(open(model_dir + "testing_label_w_subword_filter.pkl", 'rb'))
    else:
        X_test = pkl.load(open(model_dir + "testing_data_wo_subword_filter.pkl", 'rb'))
        X_test_mention = pkl.load(open(model_dir + "testing_mention_wo_subword_filter.pkl", 'rb'))
        y_test = pkl.load(open(model_dir + "testing_label_wo_subword_filter.pkl", 'rb'))

    # Training
    print("Loading training data...")
    if subword:
        X_train = pkl.load(open(model_dir + "training_data_w_subword_filter.pkl", 'rb'))
        X_train_mention = pkl.load(open(model_dir + "training_mention_w_subword_filter.pkl", 'rb'))
        y_train = pkl.load(open(model_dir + "training_label_w_subword_filter.pkl", 'rb'))
    else:
        X_train = pkl.load(open(model_dir + "training_data_wo_subword_filter.pkl", 'rb'))
        X_train_mention = pkl.load(open(model_dir + "training_mention_wo_subword_filter.pkl", 'rb'))
        y_train = pkl.load(open(model_dir + "training_label_wo_subword_filter.pkl", 'rb'))

    print("Begin training...")
    model.fit([X_train, X_train_mention], y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

    # Evaluation
    print("Loading trained weights for predicting...")
    for i in range(1,6,1):
        file = list(model_name)
        file[-6] = str(i)
        file = "".join(file)
        model.load_weights(file)
        print("Predicting...",file)
        y_pred = model.predict([X_test, X_test_mention])
    
        y_pred[y_pred >= 0.5] = 1.
        y_pred[y_pred < 0.5] = 0.
        y_pred = sparse.csr_matrix(y_pred)
    
        eval_types = ['micro','macro','weighted']
        for eval_type in eval_types:
            p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average=eval_type)
            print("[{}]\t{:3.3f}\t{:3.3f}\t{:3.3f}".format(eval_type, p, r, f))


    K.clear_session()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", action="store_true", help="Use pretrained embedding Model or not")
    parser.add_argument("--emb", help="please provide pretrained Embedding Model.")
    parser.add_argument("--subword", action="store_true" , help="Use subword or not")
    parser.add_argument("--attention",action="store_true", help="Use attention or not")
    parser.add_argument("--model", nargs='?', type=str, default="model/correct_data", 
                        help="Directory to load models. [Default: \"model/correct_data\"]")
    parser.add_argument("--mode", nargs='?', type=str, default="BLSTM",
                        help="different model architecture BLTSM or CNN [Default: \"BLSTM/\"]")
    args = parser.parse_args()

    run(args.model, args.mode, args.pre, args.emb, args.subword, args.attention)
