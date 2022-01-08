# ACD
Corpus y scripts de artículo “Automatic Cyberbullying Detection: a Mexican case in High School and Higher Education students”

En este repositorio se encuentran los archivos de datos y de secuencia de comandos utilizados en el estudio, los cuales son distribuidos como se describe a continuación.


En el archivo comprimido “cyberbullying_corpus.zip” se encuentra el corpus completo con los diálogos en lenguaje español-mexicano con presencia de cyberbullying, a que se hace referencia en el artículo. El cual consiste en lo siguiente: 
1) archivo de transcripciones de diálogos, 
2) archivo que indica la categorización de conversaciones/diálogos, 
3) archivo que indica la categorización por enunciado, 
4) carpeta con imágenes usadas en conversaciones, 
5) carpeta con imágenes tipo meme utilizadas en las conversaciones, y 
6) archivo con la transcripción del texto incluido en los archivos tipo meme.

** El archivo comprimido cuenta con contraseña, favor de solicitarla al siguiente correo electrónico aomar@uabc.edu.mx.


En el directorio “datasets” están los archivos csv usados para entrenamiento y evaluación de los modelos implementados descritos en el artículo, los cuales fueron obtenidos del corpus completo indicado en el punto anterior:
1) training_70.csv
2) test_30.csv
3) training_multivariado_70.csv
4) test_multivariado_30.csv


En el directorio “scripts_modelos” se incluyen los siguientes archivos, que contienen la secuencia de comandos para crear y evaluar los modelos descritos en el artículo:
1) El archivo Modelos_ML.py, que incluye las sentencias utilizadas para generar los modelos de clasificación usando algoritmos de aprendizaje automático.
2) Los archivos Modelos_DL-RNN.py y Modelos_DL-CNN.py, que incluyen las sentencias utilizadas para generar los modelos de clasificación usando algoritmos de aprendizaje profundo.
3) El archivo Modelos_ML-Multivariable.py, que incluye las sentencias utilizadas para generar los modelos de clasificación usando el método de Naive Bayes, usando un esquema multivariado.

