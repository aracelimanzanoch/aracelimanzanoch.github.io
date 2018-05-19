---
layout: post
title: Introducción a los sistemas de recomendación basados en filtrado colaborativo con PySpark
---

![_config.yml]({{ site.baseurl }}/images/pyspark_net.jpg)

#### Filtrado colaborativo y sistemas de recomendación con Apache Spark
Los sistemas de recomendación son ampliamente utilizados por las plataformas de servicios online y de e-commerce para solventar, de manera eficaz, el problema de sobreinformación, aportando sugerencias personalizadas a cada usuario de la plataforma, en función de sus intereses y valoraciones, así como de las de usuarios similares.
Un enfoque trivial en el desarrollo de un sistema de recomendación consistiría en sugerir a todos los usuarios los productos o servicios mejor valorados de la plataforma. Sin embargo, este enfoque no tiene en cuenta los diferentes intereses y perfiles de cada usuario ni rentabiliza aquellos productos menos populares pero con un nicho potencial mucho más específico. 
Las técnicas de Filtrado Colaborativo (Collaborative Filtering) obtienen predicciones automáticas (filtrado) sobre los intereses de cada usuario a partir de información de múltiples usuarios (colaborativo). En particular, los algoritmos basados en la  factorización implícita de la matriz de interacciones usuarios-ítems permiten desarrollar sistemas de recomendación basados en conjuntos de datos que, en el caso más básico, se componen de usuarios, ítems y valoraciones realizadas por los usuarios sobre múltiples ítems. Este caso básico es el que vamos a implementar en este artículo, utilizando la versión distribuida del algoritmo de factorización matricial con regularización, ALS (Alternating Least Squares), que ofrece la librería MLlib de Apache Spark, teniendo como objetivo clarificar los conceptos fundamentales de este modelo aplicado a la predicción de ratings.

#### Algoritmo ALS (Alternating Least Squares)
El algoritmo ALS presenta una alternativa altamente paralelizable para la optimización de la función de coste que implica la factorización de la matriz de interacciones usuarios-ítems con regularización de pesos:

![_config.yml]({{ site.baseurl }}/images/als_figure_1.jpg)

Esta alternativa se basa en el hecho de que la optimización no convexa que tiene lugar puede abordarse como dos problemas cuadráticos obtenidos fijando, de forma alternativa, los términos pu y qi, en principio desconocidos. Cuando todos los términos pu han sido fijados, el sistema optimiza los términos qi por mínimos cuadrados y vice versa ["Factorization techniques for Recommender systems" - Yehuda Koren (Yahoo Research), Robert Bell and Chris Volinsky (AT&T Labs Research), 2009].
La API de Apache Spark para Python, PySpark, en su librería MLlib ofrece una implementación paralelizada del modelo ALS para Filtrado Colaborativo que usaremos en el desarrollo del sistema de recomendación basado en predicción de ratings. Se usará la implementación correspondiente a la API basada en Dataframes de Apache Spark 2.0, la cual optimiza el rendimiento permitiendo trabajar con una abstracción de datos ampliamente conocida en la ciencia de datos, los Dataframes, de manera similar a como se haría utilizando otras librerías de Python como Pandas.
El flujo de trabajo a seguir, como en cualquier otra aplicación de Machine Learning, es el siguiente:

![_config.yml]({{ site.baseurl }}/images/config.png)

#### Dataset
El dataset utilizado para esta implementación de un sistema de recomendación con PySpark MLlib basada en predicción de ratings es el que se propone en [Brozovsky, L., Petricek, V., 2007] disponible en http://www.occamslab.com/petricek/data/. Este dataset, tal como se indica en el enlace anterior, consiste en una base de datos con 17,359,346 ratings anónimos de 168,791 perfiles realizados por 135,359 usuarios de la web de citas http://www.libimseti.cz/. Los ratings son valores enteros en una escala entre 1 y 10. Tambien se encuentra disponible informacion sobre el genero de los perfiles de usuarios de la plataforma. De esta forma, el sistema de recomendacion que se va a implementar puede describirse como una regresión de valoraciones de perfiles que trata de "rellenar" los valores vacíos en la matriz de interacciones usuarios-ítems, como aparece en este esquema:

![_config.yml]({{ site.baseurl }}/images/scheme.png)

#### Paso 1: Importar librerías e instanciar una nueva sesión Spark.
´´´ python
from __future__ import absolute_import, print_function, division
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.sql import SparkSession, Row
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, StructType, StructField, StringType
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler
# Set Spark Session as entry point
spark = SparkSession.builder\
       .appName("Simple recommendation engine using Spark MLlib")\
       .config("spark.some.config.option", "config-value")\
       .getOrCreate()\
´´´
