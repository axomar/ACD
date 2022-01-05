# -*- coding: utf-8 -*-
"""Modelos_DL-CNN.ipynb

# Algoritmos de Aprendizaje Profundo para clasificación de enunciados con presencia de cyberbullying: Redes Neuronales Convolucionales

#### Leer archivo y obtener las columnas del documento
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
training = pd.read_csv("/content/drive/.../training_70.csv")
test = pd.read_csv("/content/drive/.../test_30.csv")
list(training.columns.values)

"""#### Exploración de los datos

Para realizar algunas observaciones sobre los datos utilizamos algunas técnicas de pre-proceso de texto simples.
Como:

    Tokenizamos con NLTK. 

    Lematizamos con NLTK. [canto, cantas, canta, cantamos, cantáis, cantan son distintas formas (conjugaciones) de un mismo verbo (cantar). Y que niña, niño, niñita, niños, niñotes, y otras más, son distintas formas del vocablo niño]
    
    Transformamos los datos en minúsculas.
    
    Eliminamos stop words.
    
    
"""

import re, nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('spanish'))
wordnet_lemmatizer = WordNetLemmatizer()

def especiales(s):
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        ("ñ", "n"),
        ("√±", "n"),    
        ("\n", ""),            
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s

def normalizer(enun):

    # Elimina acentos y caracteres especiales
    enun = especiales(enun)
    
    # Eliminamos la @ y su mención
    enun = re.sub(r"@[A-Za-z0-9]+", ' ', enun)
    # Eliminamos los links de las URLs
    enun = re.sub(r"https?://[A-Za-z0-9./]+", ' ', enun)
    # Eliminamos la referencia a usuario
    enun = enun.replace('usuario', '')
    enun = enun.replace('ext', '')
    enun = enun.replace('img', '')

    ## Eliminamos todos los carácteres especiales
    only_letters = re.sub("[^a-zA-Z]", " ",enun)   
    tokens = nltk.word_tokenize(only_letters)

    # Eliminamos espacios en blanco adicionales
    enun = re.sub(r" +", ' ', enun)

    # Convertimos todo a minúsculas
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    result = " ".join(lemmas)
    return result

"""#### Observamos el resultado de este pre-proceso con una oración"""

normalizer("Ok le voy a decir que quieres invitarlo a una cita y que afortunadamente vayan a una Siguiente a Consta de tu Amiga a la que no le has dado las FLORES ")

"""#### Normalizamos el conjunto de enunciados y mostramos los primeros 5"""

training['enunciado_normalizado'] = training.transcription.apply(normalizer)
test['enunciado_normalizado'] = test.transcription.apply(normalizer)

training[['transcription','enunciado_normalizado']].head()

# Elimina enunciados con contenido vacio (en blanco)
training.drop(training[training['enunciado_normalizado'] == ""].index, inplace = True)
training.drop(training[training['enunciado_normalizado'] == " "].index, inplace = True)
count = training['enunciado_normalizado'].str.split().str.len()
training.drop(training[~(count>2)].index, inplace = True)

# Elimina enunciados con contenido vacio (en blanco)
test.drop(test[test['enunciado_normalizado'] == ""].index, inplace = True)
test.drop(test[test['enunciado_normalizado'] == " "].index, inplace = True)
count2 = test['enunciado_normalizado'].str.split().str.len()
test.drop(test[~(count2>2)].index, inplace = True)

"""# Algoritmos

## **Redes Neuronales Convolucionales**
"""

data_training = training

data_test = test

data_training.drop(["id_instance", "id_conversation", "user_code", "transcription", "detonator_dialogue"],
          axis=1,
          inplace=True)

data_test.drop(["id_instance", "id_conversation", "user_code", "transcription", "detonator_dialogue"],
          axis=1,
          inplace=True)

set(data_training.id_tone)

set(data_test.id_tone)

data_training_labels = data_training.id_tone.values

data_test_labels = data_test.id_tone.values

data_training_clean = data_training.enunciado_normalizado.values

data_test_clean = data_test.enunciado_normalizado.values

import numpy as np
import math
import re
import pandas as pd
from bs4 import BeautifulSoup

# Commented out IPython magic to ensure Python compatibility.
try:
#     %tensorflow_version 2.x
except Exception:
    pass
import tensorflow as tf

from tensorflow.keras import layers
import tensorflow_datasets as tfds

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    data_training_clean, target_vocab_size=20000)

data_training_inputs = [tokenizer.encode(sentence) for sentence in data_training_clean]

data_test_inputs = [tokenizer.encode(sentence) for sentence in data_test_clean]

MAX_LEN = max([len(sentence) for sentence in data_training_inputs])
data_training_inputs = tf.keras.preprocessing.sequence.pad_sequences(data_training_inputs,
                                                            value=0,
                                                            padding="post",
                                                            maxlen=MAX_LEN)

data_test_inputs = tf.keras.preprocessing.sequence.pad_sequences(data_test_inputs,
                                                            value=0,
                                                            padding="post",
                                                            maxlen=MAX_LEN)

training_inputs = data_training_inputs
training_labels = data_training_labels

test_inputs = data_test_inputs
test_labels = data_test_labels

class DCNN(tf.keras.Model):
    
    def __init__(self,
                 vocab_size,
                 emb_dim=128,
                 nb_filters=50,
                 FFN_units=512,
                 nb_classes=2,
                 dropout_rate=0.1,
                 training=False,
                 name="dcnn"):
        super(DCNN, self).__init__(name=name)
        
        self.embedding = layers.Embedding(vocab_size,
                                          emb_dim)
        self.bigram = layers.Conv1D(filters=nb_filters,
                                    kernel_size=2,
                                    padding="valid",
                                    activation="relu")
        self.trigram = layers.Conv1D(filters=nb_filters,
                                     kernel_size=3,
                                     padding="valid",
                                     activation="relu")
        self.pool = layers.GlobalMaxPool1D() # No tenemos variable de entrenamiento
                                             # así que podemos usar la misma capa 
                                             # para cada paso de pooling
        self.dense_1 = layers.Dense(units=FFN_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if nb_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=nb_classes,
                                           activation="softmax")
    
    def call(self, inputs, training):
        x = self.embedding(inputs)
        x_1 = self.bigram(x)
        x_1 = self.pool(x_1)
        x_2 = self.trigram(x)
        x_2 = self.pool(x_2)
        
        merged = tf.concat([x_1, x_2], axis=-1) # (batch_size, 3 * nb_filters)
        merged = self.dense_1(merged)
        merged = self.dropout(merged, training)
        output = self.last_dense(merged)
        
        return output

VOCAB_SIZE = tokenizer.vocab_size # 65540

EMB_DIM = 200
NB_FILTERS = 100
FFN_UNITS = 256
NB_CLASSES = 2  #len(set(train_labels))

DROPOUT_RATE = 0.2

BATCH_SIZE = 32
NB_EPOCHS = 5

Dcnn = DCNN(vocab_size=VOCAB_SIZE,
            emb_dim=EMB_DIM,
            nb_filters=NB_FILTERS,
            FFN_units=FFN_UNITS,
            nb_classes=NB_CLASSES,
            dropout_rate=DROPOUT_RATE)

if NB_CLASSES == 2:
    Dcnn.compile(loss="binary_crossentropy",
                 optimizer="adam",
                 metrics=["accuracy"])
else:
    Dcnn.compile(loss="sparse_categorical_crossentropy",
                 optimizer="adam",
                 metrics=["sparse_categorical_accuracy"])

checkpoint_path = "/content/drive/My Drive/Backup 2021/NLP/Tensorflow/Bullying/ckptV3/"


ckpt = tf.train.Checkpoint(Dcnn=Dcnn)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Último checkpoint restaurado!!")

Dcnn.fit(training_inputs,
         training_labels,
         batch_size=BATCH_SIZE,
         epochs=NB_EPOCHS)
ckpt_manager.save()

results = Dcnn.evaluate(test_inputs, test_labels, batch_size=BATCH_SIZE)
print(results)

from sklearn.metrics import precision_recall_fscore_support as score
import numpy as numpy

predicciones=Dcnn.predict(test_inputs)
prediction = np.asarray(predicciones.round())

precision, recall, fscore, support = score(test_labels, prediction)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))