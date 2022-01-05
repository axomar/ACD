# -*- coding: utf-8 -*-
"""Modelos_DL-RNN.ipynb

# Algoritmos de Aprendizaje Profundo para clasificación de enunciados con presencia de cyberbullying: Redes Neuronales Recurrentes

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
training.drop(training[~(count>1)].index, inplace = True)

# Elimina enunciados con contenido vacio (en blanco)
test.drop(test[test['enunciado_normalizado'] == ""].index, inplace = True)
test.drop(test[test['enunciado_normalizado'] == " "].index, inplace = True)
count2 = test['enunciado_normalizado'].str.split().str.len()
test.drop(test[~(count2>1)].index, inplace = True)

"""# Algoritmos

## **Redes Neuronales Recurrentes**
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

import tensorflow as tf
import numpy as np

tf.__version__

dataframe_training = training[['enunciado_normalizado','id_tone']]
dataframe_test = test[['enunciado_normalizado','id_tone']]

X_training = dataframe_training['enunciado_normalizado']
y_training = dataframe_training['id_tone'].astype(np.uint8)
X_test = dataframe_test['enunciado_normalizado']
y_test = dataframe_test['id_tone'].astype(np.uint8)

from keras.preprocessing.text import Tokenizer

NUMBER_OF_WORDS = 2200
MAX_LEN = 50

#tokenizer = Tokenizer(num_words = NUMBER_OF_WORDS)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_training)

X_training = tokenizer.texts_to_sequences(X_training)
X_test = tokenizer.texts_to_sequences(X_test)

from keras.preprocessing.sequence import pad_sequences

X_training = pad_sequences(X_training, padding='post', maxlen=MAX_LEN)
X_test = pad_sequences(X_test, padding='post', maxlen=MAX_LEN)

VOCABULARY_SIZE = NUMBER_OF_WORDS
EMBEDDING_SIZE = 100

### Elaboración de capas

model = tf.keras.Sequential()

## CAPA DE EMBEDDING (CAPA DE INCRUSTRACIÓN), CAPA DONDE SUMINISTRAMOS LOS TEXTOS
## SE UTILIZA PARA CREAR UN VECTOR DE PALABRAS DE UNA VALORACIÓN DETERMINADA (VECTOR DE PALABRAS WORD2VEC)
## TRANSFORMA EL TEXTO A FORMATO NUMÉRICO (MATRIZ MUY GRANDE: CADA FILA ES UNA VALORACIÓN Y LAS COLUMNAS SON LAS PALABRAS)

model.add(tf.keras.layers.Embedding(VOCABULARY_SIZE, 
                                    EMBEDDING_SIZE, 
                                    input_shape=(X_training.shape[1],)))


model.add(tf.keras.layers.LSTM(units=100, activation='tanh'))


#En units, se especifica cuantas clases tenemos. 
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Entrenamiento del modelo usando 10 epocas
model.fit(X_training, y_training, epochs=10, batch_size=10)

# Evaluación ed accuracy
test_loss, test_acurracy = model.evaluate(X_test, y_test)
print("Test accuracy: {}".format(test_acurracy))

# Estimación del resto de métricas de desempeño
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as numpy

predicciones=model.predict(X_test)
prediction = np.asarray(predicciones.round())

precision, recall, fscore, support = score(y_test, prediction)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))