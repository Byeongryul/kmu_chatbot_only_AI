from transformers import (BertConfig, BertTokenizer, TFBertModel)
import numpy as np
import pandas as pd
import tensorflow as tf

class Bert:   
    def __init__(self, fname = 'test.csv', weight = 'dataset_v4.h5'):
        self.fname = fname
        self.weight = weight
        self.idx2label_k = {}
        self.num_labels = 0
        self.model = tf.keras.models.Model()
        self.idx2label()
        self.configBert()
        self.modeling()

    def idx2label(self):
        df = pd.read_csv('rsc/preprocessed_data/' + self.fname)
        tag_list = np.sort(df.Tag.unique())[::-1]
        label_map = {label: i+1 for i, label in enumerate(tag_list)}   ##tag_list에 레이블 
        idx2label = {i: w for w, i in label_map.items()}  ##tag_list을 뒤집어 놓은거
        idx2label[0] = 'Null'
        self.idx2label_k = idx2label
        self.num_labels = len(tag_list)+1
        for key, values in idx2label.items():
            if '-' in idx2label[key]:
                self.idx2label_k[key] = values.split('-')[1]

    def configBert(self):
        ''' BERT 모델 설정 '''
        self.max_seq_length =128
        self.pad_token=0
        self.pad_token_segment_id=0
        self.sequence_a_segment_id=0
        self.pad_token_label_id = 0
        self.BERT_MODEL="bert-base-multilingual-cased"
        self.config = BertConfig.from_pretrained(self.BERT_MODEL)     # transformer BERT 라이브러리 사용을 위한 config
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL,do_lower_case=False)    # transformer BERT 라이브러리 중 Tokenizer

    def modeling(self):
        ''' 신경망 모델 작업 현재 모델 model 3 '''
        MAX_LENGTH=self.max_seq_length   ## 위 bert 입력값 만들때 사용한 값

        ## BERT 레이어 만들기
        bert_model = TFBertModel.from_pretrained(self.BERT_MODEL, config= self.config)
        bert_model.trainable = True     ## BERT 파인튜닝 할 시 True
        ## BERT 입력값 형식 맞춰주기
        input_ids = tf.keras.Input(shape=(MAX_LENGTH,), dtype=tf.int32,name="input_ids"   )
        attention_masks = tf.keras.Input(shape=(MAX_LENGTH,),dtype=tf.int32,name="attention_masks"   )
        token_type_ids = tf.keras.Input(shape=(MAX_LENGTH,), dtype=tf.int32,name="token_type_ids"  )

        ## 신경망 레이서 쌓기 시작
        sequence_output, pooled_output = bert_model(input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids)

        bi_lstm = tf.keras.layers.Bidirectional( tf.keras.layers.LSTM(64*4, return_sequences=True, recurrent_dropout=0.1) )(sequence_output)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.num_labels, activation = "softmax"))(bi_lstm)

        model = tf.keras.models.Model(  inputs=[input_ids, attention_masks, token_type_ids], outputs=output )

        model.compile( optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="categorical_crossentropy", metrics=["accuracy"] )

        #model.summary()
        model.load_weights('rsc/pretrained_model/' + self.weight)
        self.model =  model