# -*- coding: utf-8 -*-
"""Modelos_ML.ipynb

# Algoritmos de Aprendizaje Automático para clasificación de enunciados con presencia de cyberbullying

#### Leer archivo y desplegar el nombre de las columnas del documento
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

"""#### Normalizamos el conjunto de enunciados de los datos de entrenamiento y prueba, y mostramos los primeros 5 de entrenamiento"""

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

vectorized_training = count_vectorizer.fit_transform(training.enunciado_normalizado)
indexed_training = hstack((np.array(range(0,vectorized_training.shape[0]))[:,None], vectorized_training))

vectorized_test = count_vectorizer.transform(test.enunciado_normalizado)
indexed_test = hstack((np.array(range(0,vectorized_test.shape[0]))[:,None], vectorized_test))

# Generamos vector de etiquetas para cada dataset (entrenamiento y prueba)
def sentiment2target(sentiment):
    return {
        0: 0,
        1: 1,
        2: 2,
        3: 3
    }[sentiment]
targets_training = training.id_tone.apply(sentiment2target)

targets_test = test.id_tone.apply(sentiment2target)

"""###  Utilizamos un 70% de los enunciados para entrenamiento y 30% para pruebas

Se utilizan 2 archivos predefinidos donde se han seleccionado aleatoriamente un conjunto de enunciados para el 70% que se utulizara en el enternamiento y el 30% que se utilizará para pruebas con todos los algoritmos de ML utilizados. 
"""

from sklearn.model_selection import train_test_split
data_training_index = indexed_training.tocsr()[:,0]

data_training = indexed_training.tocsr()[:,1:]

data_test_index = indexed_test.tocsr()[:,0]
data_test = indexed_test.tocsr()[:,1:]

"""## Pasamos los datos a los distintos clasificadores

# Support Vector Machine (SVM)
Usamos OneVsRestClassifier. 
Esto nos permite obtener la distribución de probabilidad en las diferentes etiquetas o clases. 
Detrás de escena, en realidad creamos clasificadores segúnn el número de etiquetas(clases). 
Cada uno de estos clasificadores determina la probabilidad de que el punto de datos pertenezca a su clase correspondiente, o cualquiera de las otras clases.
"""

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
clf = svm.SVC(gamma=0.01, C=100., probability=True, class_weight='balanced', kernel='linear')
clf_output = clf.fit(data_training, targets_training)

"""## Evaluación de resultados
Obtenemos la precisión media en los datos de prueba y las etiquetas dadas.
"""

clf.score(data_test, targets_test)

# Obtenemos el desempeño del modelo usando diferentes métricas.
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as numpy

predicciones=clf.predict(data_test)
precision, recall, fscore, support = score(targets_test, predicciones)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

"""# Random Forest Classifier"""

#Clasificador RandomForest
from sklearn.ensemble import RandomForestClassifier

clf2 = RandomForestClassifier(n_estimators=10000, random_state=0)

clf_output = clf2.fit(data_training, targets_training)

"""### Evaluación de resultados"""

clf2.score(data_test, targets_test)

# Obtenemos el desempeño del modelo usando diferentes métricas.
from sklearn.metrics import precision_recall_fscore_support as score
import numpy as numpy

predicciones=clf2.predict(data_test)
precision, recall, fscore, support = score(targets_test, predicciones)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

"""# Naive Bayes"""

#Clasificador Naive Bayes multinomial
from sklearn.naive_bayes import MultinomialNB
clf3 = MultinomialNB()
clf_output = clf3.fit(data_training, targets_training)

"""### Evaluación de resultados"""

clf3.score(data_test, targets_test)

from sklearn.metrics import precision_recall_fscore_support as score
import numpy as numpy

predicciones=clf3.predict(data_test)
precision, recall, fscore, support = score(targets_test, predicciones)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

"""# Decision tree Classifier"""

#Clasificador árboles de decisión
from sklearn.tree import DecisionTreeClassifier 
clf4 = DecisionTreeClassifier()

clf_output = clf4.fit(data_training, targets_training)

"""### Evaluación de resultados"""

clf4.score(data_test, targets_test)

from sklearn.metrics import precision_recall_fscore_support as score
import numpy as numpy

predicciones=clf4.predict(data_test)
precision, recall, fscore, support = score(targets_test, predicciones)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))

"""**Logistic Regression**"""

from sklearn.linear_model import LogisticRegression

clf5 = LogisticRegression(random_state=0, max_iter=100)
clf_output = clf5.fit(data_training, targets_training)

clf5.score(data_test, targets_test)

from sklearn.metrics import precision_recall_fscore_support as score
import numpy as numpy

predicciones=clf5.predict(data_test)
precision, recall, fscore, support = score(targets_test, predicciones)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))