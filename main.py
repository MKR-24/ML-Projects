import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM,Dense,Activation,Dropout
from keras.optimizers import Adam
from keras.models import load_model
filepath=tf.keras.utils.get_file('shakesphere.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text=open(filepath,'rb').read().decode(encoding='utf-8').lower()

text=text[300000:800000]

characters = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_char = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
STEP_SIZE = 3
sentences=[]
next_characters=[]

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH,len(characters)), dtype=bool)
y = np.zeros((len(sentences),len(characters)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

model=Sequential()
model.add(LSTM(256,input_shape=(SEQ_LENGTH,len(characters)), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

optimizer = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.fit(x,y,batch_size=64,epochs=50)
model.save('RNN_PoeticTexts.model')
model=load_model('RNN_PoeticTexts.model')
# def sample(preds, temperature=1.0):
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)

# def generate_text(length,temperature):
#   start_index=random.randint(0,len(text)-SEQ_LENGTH-1)
#   generated_text=''
#   sentence=text[start_index:start_index+SEQ_LENGTH]
#   generated_text+=sentence
#   for i in range(length):
#     x=np.zeros((1,SEQ_LENGTH,len(characters)))
#     for t,character in enumerate(sentence):
#       x[0,t,char_to_index[character]]=1

#     predictions=model.predict(x,verbose=0)[0]
#     next_index=sample(predictions,temperature)
#     next_character=index_to_char[next_index]
#     generated_text+=next_character
#     senetence=sentence[1:]+next_character
#   return generated_text

# print('----0.2----')
# print(generate_text(10,0.2))
# print('----0.4----')
# print(generate_text(10,0.4))
# print('----0.6----')
# print(generate_text(10,0.6))
# print('----0.8----')
# print(generate_text(10,0.8 ))
# print('----1----')
# print(generate_text(10,1.0))