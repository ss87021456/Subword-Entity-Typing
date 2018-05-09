from keras.models import Model
from keras.layers import Dense, Embedding, Input, concatenate, dot, Permute, Reshape, merge
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.layers import Conv1D, Conv2D, MaxPooling1D


def attention_3d_block(inputs):
    SINGLE_ATTENTION_VECTOR = False
    MAX_SEQUENCE_LENGTH = 40
    EMBEDDING_DIM = 100
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((EMBEDDING_DIM, MAX_SEQUENCE_LENGTH))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(MAX_SEQUENCE_LENGTH, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(EMBEDDING_DIM)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

def BLSTM(label_num, sentence_emb=None, mention_emb=None, attention=False, mode='concatenate', dropout=0.1, subword=False):
        
        MAX_NUM_WORDS = 30000
        MAX_NUM_MENTION_WORDS = 20000
        MAX_SEQUENCE_LENGTH = 40
        if subword:
            MAX_MENTION_LENGTH = 15
        else:
            MAX_MENTION_LENGTH = 5
        EMBEDDING_DIM = 100

        sentence = Input(shape=(MAX_SEQUENCE_LENGTH, ), name='sentence')
        
        # Pretrain sentence_embedding
        if sentence_emb is not None:
            x = sentence_emb(sentence)
        else:
            x = Embedding(MAX_NUM_WORDS,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH)(sentence)
        
        if attention: # attention before lstm
            x = attention_3d_block(x)

        x = Bidirectional(LSTM(50, return_sequences=True))(x)
        x = GlobalMaxPool1D()(x)

        mention = Input(shape=(MAX_MENTION_LENGTH, ), name='mention')
        # Pretrain mention_embedding
        if mention_emb is not None:
            x_2 = mention_emb(mention)
        else:
            x_2 = Embedding(MAX_NUM_MENTION_WORDS,EMBEDDING_DIM,input_length=MAX_MENTION_LENGTH)(mention)
        
        x_2 = Bidirectional(LSTM(50, return_sequences=True))(x_2)
        x_2 = GlobalMaxPool1D()(x_2)

        if mode == 'concatenate':
            x = concatenate([x, x_2])           # Concatencate
            x = Dropout(dropout)(x)
        elif mode == 'dot':
            x = dot([x, x_2], axes=-1)           # Dot product

        x = Dense(200, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = Dense(label_num, activation="sigmoid")(x)
        model = Model(inputs=[sentence, mention], outputs=x)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

def CNN(label_num, sentence_emb=None, mention_emb=None, attention=False, mode='concatenate', dropout=0.1, subword=False):
        
        MAX_NUM_WORDS = 30000
        MAX_NUM_MENTION_WORDS = 20000
        MAX_SEQUENCE_LENGTH = 40
        if subword:
            MAX_MENTION_LENGTH = 15
        else:
            MAX_MENTION_LENGTH = 5
        EMBEDDING_DIM = 100
        num_filters = 64

        sentence = Input(shape=(MAX_SEQUENCE_LENGTH, ), name='sentence')        
        
        # Pretrain sentence_embedding
        if sentence_emb is not None:
            x = sentence_emb(sentence)
        else:
            x = Embedding(MAX_NUM_WORDS,EMBEDDING_DIM,input_length=MAX_SEQUENCE_LENGTH)(sentence)
        
        if attention: # attention before lstm
            x = attention_3d_block(x)

        x = Conv1D(num_filters, 5, activation='relu', padding='valid')(x)
        x = MaxPooling1D(2)(x)
        x = Conv1D(num_filters, 5, activation='relu', padding='valid')(x)
        x = GlobalMaxPool1D()(x)

        mention = Input(shape=(MAX_MENTION_LENGTH, ), name='mention')
        # Pretrain mention_embedding
        if mention_emb is not None:
            x_2 = mention_emb(mention)
        else:
            x_2 = Embedding(MAX_NUM_MENTION_WORDS,EMBEDDING_DIM,input_length=MAX_MENTION_LENGTH)(mention)

        x_2 = Conv1D(num_filters, 5, activation='relu', padding='same')(x_2)
        x_2 = MaxPooling1D(2)(x_2)
        x_2 = Conv1D(num_filters, 5, activation='relu', padding='same')(x_2)
        x_2 = GlobalMaxPool1D()(x_2)

        if mode == 'concatenate':
            x = concatenate([x, x_2])           # Concatencate
            x = Dropout(dropout)(x)
        elif mode == 'dot':
            x = dot([x, x_2], axes=-1)           # Dot product

        x = Dense(200, activation="relu")(x)
        x = Dropout(dropout)(x)
        x = Dense(label_num, activation="sigmoid")(x)
        model = Model(inputs=[sentence, mention], outputs=x)
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model