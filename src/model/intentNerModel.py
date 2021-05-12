from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from src.model.bert import Bert
import numpy as np
import logging
logging.basicConfig(level=logging.ERROR)

class IntentNerModel:
  def __init__(self):
    self.bert = Bert()
    self.tokenizer = BertTokenizer.from_pretrained(self.bert.BERT_MODEL,do_lower_case=False)
    self.cs_no = 0.7

  def input2intentNer(self, sentence) :
    #버트 입력값 만들기
    input_ids_sentence, token_ids_sentence, attention_masks_sentence = self.convert_sentences_to_input(sentence)
    max_seq_length = self.bert.max_seq_length
    #버트 형식에 맞게 패딩작업
    input_ids_sentence = pad_sequences(input_ids_sentence,maxlen=max_seq_length,dtype="long",truncating="post",padding="post")
    token_ids_sentence = pad_sequences(token_ids_sentence,maxlen=max_seq_length,dtype="long",truncating="post",padding="post")
    attention_masks_sentence = pad_sequences(attention_masks_sentence,maxlen=max_seq_length,dtype="long",truncating="post",padding="post")
    
    bert_input = [input_ids_sentence, token_ids_sentence,attention_masks_sentence]
    ner_score = self.bert.model.predict(bert_input)
    ner_output = np.argmax(ner_score, axis=-1)
    
    lists = []
    for i in range(len(input_ids_sentence[0])) :
      piece = ner_output[0][i]
      if input_ids_sentence[0][i]!=101 and input_ids_sentence[0][i]!=102 and input_ids_sentence[0][i]!=0 : 
        word = self.tokenizer.decode(input_ids_sentence[0][i:i+1])
        if word[0:2]!="##" :
          if len(lists) != 0 and lists[-1][list(lists[-1].keys())[0]] == '':
            lists[-1][list(lists[-1].keys())[0]] = words
          if piece > 1.5 and max(ner_score[0][i]) >= self.cs_no:
            lists.append({self.bert.idx2label_k.get(piece):''})
          words = word
        else :
          words += word[2:]
    if lists[-1][list(lists[-1].keys())[0]] == '':
      lists[-1][list(lists[-1].keys())[0]] = words
    return lists

  def convert_sentences_to_input(self, sentences):
    tokens = []
    max_seq_length = self.bert.max_seq_length

    word_tokens = self.tokenizer.tokenize('intent ' + sentences)
    tokens.extend(word_tokens)
  
    special_tokens_count =  2
    # 길면 자름
    if len(tokens) > max_seq_length - special_tokens_count:
      tokens = tokens[: (max_seq_length - special_tokens_count)]
    # 빈 공간 채움
    inputs = self.tokenizer.encode_plus(tokens,add_special_tokens=True, max_length=max_seq_length)
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
    attention_masks = [1] * len(input_ids)

    return [input_ids], [token_type_ids], [attention_masks]