# -*- coding: utf-8 -*-
"""Modelos_ML_Multivariable.ipynb

# Algoritmos de Aprendizaje Automático para clasificación de enunciados con presencia de cyberbullying, utilizando análisis multivariado

#### Leer archivo y desplegar el nombre de las columnas del documento
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
training = pd.read_csv("/content/drive/.../training_multivariado_70", encoding= 'unicode_escape')
test = pd.read_csv("/content/drive/.../test_multivariado_30", encoding= 'unicode_escape')
list(training.columns.values)

"""#### Devolvemos las 5 filas superiores del dataset"""

training.head()

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

training.head()

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

training.head()

"""# Algoritmos

### Preparando los datos

El clasificador tendrá en cuenta cada palabra única presente en la oración, así como todas las palabras consecutivas. 

Para que esta representación sea útil para nuestro clasificador, transformamos cada oración en un vector. 

El vector tiene la misma longitud que nuestro vocabulario, es decir, la lista de todas las palabras observadas en nuestros datos de entrenamiento, y cada palabra representa una entrada en el vector. 
Si una palabra en particular está presente, esa entrada en el vector es 1, de lo contrario 0.

Para crear estos vectores utilizamos el CountVectorizer de sklearn.

CountVectorizer implementa la tokenización como el recuento de ocurrencias.
"""

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
count_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2))

def sentiment2target(sentiment):
    return {
        0: 0,
        1: 1,
        2: 2,
        3: 3
    }[sentiment]
#targets = enunciado.Bullying.apply(sentiment2target)
targets_training = pd.Series(training.Bullying)

#targets2 = enunciado2.Bullying.apply(sentiment2target)
targets_test = pd.Series(test.Bullying)

"""USANDO MULTIPLE FEATURES UNIDAS CON ColumnTransformer"""

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

## Descomente la variante de características que se desea evaluar

# Features: text
##column_trans = ColumnTransformer([('text', CountVectorizer(analyzer='word', ngram_range=(1,2)), 'transcription')], remainder='drop', verbose=False, sparse_threshold=0)
##new_vectorized_training = np.array(column_trans.fit_transform(training))
## new_vectorized_test = np.array(column_trans.transform(test))

# Features: text y palabras unicas (frecuencia 1)
##column_trans = ColumnTransformer([('PU', SimpleImputer(), ['Palabra Unica']),('text', CountVectorizer(analyzer='word', ngram_range=(1,2)), 'transcription')], remainder='drop', verbose=False, sparse_threshold=0)
##new_vectorized_training = np.array(column_trans.fit_transform(training))
##new_vectorized_test = np.array(column_trans.transform(test))

# Features: text y cantidad de participantes en la conversacion
##column_trans = ColumnTransformer([('P', SimpleImputer(), ['Total']),('text', CountVectorizer(analyzer='word', ngram_range=(1,2)), 'transcription')], remainder='drop', verbose=False, sparse_threshold=0)
##new_vectorized_training = np.array(column_trans.fit_transform(training))
##new_vectorized_test = np.array(column_trans.transform(test))

# Features: text y enunciado detonador
##column_trans = ColumnTransformer([('D', SimpleImputer(), ['detonator_dialogue']),('text', CountVectorizer(analyzer='word', ngram_range=(1,2)), 'transcription')], remainder='drop', verbose=False, sparse_threshold=0)
##new_vectorized_training = np.array(column_trans.fit_transform(training))
##new_vectorized_test = np.array(column_trans.transform(test))

# Features: text y enunciado detonador y palabra unica
column_trans = ColumnTransformer([('D', SimpleImputer(), ['detonator_dialogue']),('PU', SimpleImputer(), ['Palabra Unica']),('text', CountVectorizer(analyzer='word', ngram_range=(1,2)), 'transcription')], remainder='drop', verbose=False, sparse_threshold=0)
new_vectorized_training = np.array(column_trans.fit_transform(training))
new_vectorized_test = np.array(column_trans.transform(test))

# Features: text y enunciado detonador, palabra unica y participantes
##column_trans = ColumnTransformer([('D', SimpleImputer(), ['detonator_dialogue']),('PU', SimpleImputer(), ['Palabra Unica']),('P', SimpleImputer(), ['Total']),('text', CountVectorizer(analyzer='word', ngram_range=(1,2)), 'transcription')], remainder='drop', verbose=False, sparse_threshold=0)
##new_vectorized_training = np.array(column_trans.fit_transform(training))
##new_vectorized_test = np.array(column_trans.transform(test))

# Features: text y total de participantes- palabras unicas
##column_trans = ColumnTransformer([('PU', SimpleImputer(), ['Palabra Unica']),('P', SimpleImputer(), ['Total']),('text', CountVectorizer(analyzer='word', ngram_range=(1,2)), 'transcription')], remainder='drop', verbose=False, sparse_threshold=0)
##new_vectorized_training = np.array(column_trans.fit_transform(training))
##new_vectorized_test = np.array(column_trans.transform(test))

# Features: text y total de participantes
##column_trans = ColumnTransformer([('PU', OneHotEncoder(dtype='int'), ['Palabra Unica']),('Total', OneHotEncoder(dtype='int'), ['Total']),('text', CountVectorizer(analyzer='word', ngram_range=(1,2)), 'enunciado_normalizado')], remainder='drop', verbose=False, sparse_threshold=0)
##new_vectorized_training = np.array(column_trans.fit_transform(training))
##new_vectorized_test = np.array(column_trans.transform(test))

from scipy import sparse
Snew_vectorized_training = sparse.csr_matrix(new_vectorized_training)
Snew_indexed_training = hstack((np.array(range(0,Snew_vectorized_training.shape[0]))[:,None], Snew_vectorized_training))

Snew_vectorized_test = sparse.csr_matrix(new_vectorized_test)
Snew_indexed_test = hstack((np.array(range(0,Snew_vectorized_test.shape[0]))[:,None], Snew_vectorized_test))

new_data_training_index = Snew_indexed_training.tocsr()[:,0]
new_data_training = Snew_indexed_training.tocsr()[:,1:]

new_data_test_index = Snew_indexed_test.tocsr()[:,0]
new_data_test = Snew_indexed_test.tocsr()[:,1:]

"""###  Utilizamos un 70% de los enunciados para entrenamiento y 30% para pruebas

Se utilizan 2 archivos predefinidos donde se han seleccionado aleatoriamente un conjunto de enunciados para el 70% que se utulizara en el enternamiento y el 30% que se utilizará para pruebas.

# Naive Bayes
"""

#Clasificador Naive Bayes multinomial
from sklearn.naive_bayes import MultinomialNB
clf3 = MultinomialNB()
clf_output = clf3.fit(new_data_training, targets_training)

"""### Evaluación de resultados"""

clf3.score(new_data_test, targets_test)

from sklearn.metrics import precision_recall_fscore_support as score
import numpy as numpy

predicciones=clf3.predict(new_data_test)
precision, recall, fscore, support = score(targets_test, predicciones)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))