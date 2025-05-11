# ***Redes neuronales convolucionales para detectar imágenes sintéticas***

## *Evander Gila Reques, TFG de Ing. de Software*

Este proyecto se fundamenta en el creciente auge de las IAs generativas en un mundo global cada vez más interconectado, utilizadas en ocasiones para la creación de *"fake news"*. Actualmente la calidad de estas imágenes generadas sintéticamente ha avanzado a pasos agigantados, siendo muy difíciles de reconocer y distinguir de una imagen real.
Para ello, en este trabajo se han diseñado y desarrollado tres arquitecturas de CNNs diferentes diseñadas para discernir entre imágenes reales e imágenes sintéticamente generadas, y se han entrenado varios modelos variando arquitecturas e hiperparámetros tales como el *learning rate* o el uso de DA (*Data Augmentation*).
Una vez se han entrenado y validado todas las variaciones de modelos posibles, se han extraido unas métricas mediante el proceso de test de estos modelos, puediendo compararlos entre sí y evaluar cuál es el más óptimo para desarrollar la tarea de discriminación.

### *Dataset*

Para el entrenamiento de estos modelos, se ha usado el *dataset "CIFAKE: Real and AI-Generated Synthetic Images"* (https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images), el cual está diseñado específicamente para abordar este tipo de problemas, conteniendo 120K imágenes, siendo 60K reales y 60K generadas sintéticamente. Las imágenes destinadas al entrenamiento serían 100K y a test serían 20K. Las imágenes reales proceden del *dataset CIFAR-10*, con imágenes a color de 32x32 píxeles distribuidas en 10 clases. Las imágenes sintéticas se generaron mediante *Stable Diffusion v1.4*. Por tanto, este *dataset* ofrece una báse sólida para el entrenamiento de los modelos de este proyecto.

### *Entrenamiento y validación*

En este proceso se ha efectuado el entrenamiento de 20 modelos distintos que comparten el objetivo de discriminar imágenes reales y generadas sintéticamente.
Los pasos seguidos para entrenar estos 20 modelos son los siguientes:

#### *Arquitectura*
Se han diseñado tres arquitecturas de redes convolucionales distintas, basándose en el número de capas de la arquitectura y sus canales de cada capa. Las arquitecturas se encuentran definidas en el archivo *architecture.py* y son las siguientes:

-***Arquitectura 3C:*** Contiene 3 capas convolucionales con 64, 128 y 256 canales respectivamente.

-***Arquitectura 4C:*** Contiene 3 capas convolucionales con 64, 128, 256 y 512 canales respectivamente.

-***Arquitectura B3C:*** Contiene 3 bloques de 3 capas convolucionales con 64, 128 y 256 canales en el primer, segundo y tercer bloque respectivamente.

Después de cada capa convolucional se normalizan las activaciones de cada capa, se aplica *ReLu* y se realiza un *MaxPooling* (A excepción de la arquitectura B3C que aplica este último al final de cada bloque).
Después se aplana el tensor producido, se aplica una primera capa de *fully connected layer*, se aplica un *dropout* del 0.5 para evitar el *overfitting* y se aplica una segunda capa de *fully connected layer* con la neurona final.

#### *Preprocesado de los datos*

Para poder utilizar los datos del *dataset*, se han dividido los datos de entrenamiento en dos, entrenamiento y validación, en una proporción 70/30. Dejando así 70K *(35K reales-35K falsas)* imágenes para entrenamiento y 30K *(15K reales-15K falsas)* para validación.
Se ha aplicado una normalización necesaria de estos datos para su procesamiento, siendo todas redimensionadas de 32x32 a 64x64, transformadas a tensor y normalizadas según los valores de la media y desviación estándar del propio dataset calculados mediante la función de *get_media_std_training(Data_dir)*.

Se han creado dos *DataLoaders* distintos, en función de si utilizan DA o no, para cada conjunto de datos (entrenamiento y validación). Los que utilizan DA aplican *RandAugment* para automatizar este proceso de randomización.

Todos estos métodos se encuentran en el archivo *preprocess.py* .
La comprobación de la correción de los datos y su asignación se encuentra en el archivo *Comprobaciones.ipynb*
#### *Variables*

Los modelos entrenados se distinguen entre ellos por la combinación de tres "variables", la arquitectura, el *learning rate* y el uso del DA.
La variable de la arquitectura puede tomar tres valores: 3C, 4C y B3C.
La variable del learning rate puede tomar 3 valores: 0.001, 0.0001 y 0.0002.
La variable del DA puede tomar dos valores: True y False.

Esto da un total de 18 combinaciones de modelos distintas a entrenar. El resto de hiperparámetros como el número de epochs (20), el tamaño de la imagen (64x64), el *batch_size* (32) o el tamaño del kernel (3x3) se quedan estáticas.

(Solo hay dos excepciones en segundas versiones de modelos de arquitecturas 3C y 4C que se entrenan a 30 epochs porque su mejor versión de validación se encontraba en el  epoch 20 y había sospechas de mejora, dejando 20 modelos entrenados)

Todos los hiperparámetros y variables se establecen en el archivo *Entrenamiento.ipnb* .
#### *Desarrollo del entrenamiento y validación*

Para el entrenamiento y validación se han utilizado las siguientes funciones:

.Optimizador: Adam
.Función de pérdida: BCEWithLogitsLoss (utiliza sigmoid)
.Función de precisión: Personalizada (utilizando sigmoid)

El proceso de entrenamiento y validación se ha realizado conjuntamente. Simultáneamente a este proceso, se han recogido una métricas de entrenamiento y validación, las cuales se encuentran en el directorio de "Entrenamiento".
En esta carpeta se encuentran los 20 modelos en formato *.pth*, las 20 gráficas de entrenamiento y validación de la pérdida y la precisión y las matrices de confusión. También se encuentra un csv con los resultados del entrenamiento y validación de todos los modelos.

El código del entrenamiento se encuentra en *entrenamiento.py* y está gestionado en el archivo *Entrenamiento.ipynb* .
El código para la generación de gráficos y la muestra en pantalla de imágenes y matrices se encuentra en *display.py* .

### *Test*

Para el desarrollo de los tests de los modelos entrenados y validados se han utilizado el *DataLoader* pertinente de la clase *preprocess.py* y se han automatizado todos los test de forma que se puedan pasar por arquitecturas y todos a la vez.

Para el análisis de los modelos se han sacado 4 métricas: TPR, FPR, Precisión y AUC.
Todas ellas se han almacenado en csv y se ha nsacado las matrices de confusión del proceso de test junto con las curvas ROC y las AUC.

El código del proceso  de test se encuentra en el archivo *testt.py* y es gestionado en el archivo *Test.ipynb*
Para la muestra de los gráficos se ha recurrido al código del archivo *display*


### *Dependencias*
Se han utilizado los paquetes de:

- Pytorch
- matplotlib
- pandas
- sklearn.metrics
- seaborn
- numpy
- csv
- os


###### Este proyecto constituye una base para el segundo proyecto de fin de grado y una base sólida para la creación de otras redes neuronales con el mismo propósito.


