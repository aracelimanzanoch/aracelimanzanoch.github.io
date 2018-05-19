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

![_config.yml]({{ site.baseurl }}/images/ml_workflow.jpg)

#### Dataset
El dataset utilizado para esta implementación de un sistema de recomendación con PySpark MLlib basada en predicción de ratings es el que se propone en [Brozovsky, L., Petricek, V., 2007] disponible en http://www.occamslab.com/petricek/data/. Este dataset, tal como se indica en el enlace anterior, consiste en una base de datos con 17,359,346 ratings anónimos de 168,791 perfiles realizados por 135,359 usuarios de la web de citas http://www.libimseti.cz/. Los ratings son valores enteros en una escala entre 1 y 10. Tambien se encuentra disponible informacion sobre el genero de los perfiles de usuarios de la plataforma. De esta forma, el sistema de recomendacion que se va a implementar puede describirse como una regresión de valoraciones de perfiles que trata de "rellenar" los valores vacíos en la matriz de interacciones usuarios-ítems, como aparece en este esquema:

![_config.yml]({{ site.baseurl }}/images/scheme.png)

#### Paso 1: Importar librerías e instanciar una nueva sesión Spark.
```python
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
```
#### Paso 2: Cargar el dataset en un dataframe de Spark, limpiar las filas con valores vacíos y normalizar los datos de ratings.

```python
# Parse dating agency ratings data as a Spark dataframe
ratings = "ratings.dat"
schema = StructType([StructField("user_id", IntegerType(), False),
                     StructField("profile_id", IntegerType(), False),
                     StructField("rating", IntegerType(), True)])
ratings_df = spark.read.format("csv").option("header", "false").option("delimiter", ",").schema(schema).load(ratings)
ratings_df = ratings_df.na.drop(how="any")
ratings_df.show(3)
ratings_df.printSchema()
------------------------------------------------------------------
+-------+----------+------+
|user_id|profile_id|rating|
+-------+----------+------+
|      1|       133|     8|
|      1|       720|     6|
|      1|       971|    10|
+-------+----------+------+
only showing top 3 rows

root
 |-- user_id: integer (nullable = true)
 |-- profile_id: integer (nullable = true)
 |-- rating: integer (nullable = true)
------------------------------------------------------------------
# Parse profiles gender data as a Spark dataframe
gender_data = "gender.dat"
schema = StructType([StructField("profile_id", IntegerType(), False),
                     StructField("gender", StringType(), False)])
gender_df = spark.read.format("csv").option("header", "false").option("delimiter", ",").schema(schema).load(gender_data)
gender_df.show(3)
------------------------------------------------------------------
+----------+------+
|profile_id|gender|
+----------+------+
|         1|     F|
|         2|     F|
|         3|     U|
+----------+------+
only showing top 3 rows
------------------------------------------------------------------
# Normalize rating column
min_rating = 1
max_rating = 10
ratings_df = ratings_df.withColumn('norm_rating', (ratings_df.rating-min_rating)/(max_rating-min_rating))
df = ratings_df\
     .select("user_id","profile_id","norm_rating")\
     .withColumnRenamed("norm_rating", "label")
df.show(5)
------------------------------------------------------------------
+-------+----------+------------------+
|user_id|profile_id|             label|
+-------+----------+------------------+
|      1|       133|0.7777777777777778|
|      1|       720|0.5555555555555556|
|      1|       971|               1.0|
|      1|      1095|0.6666666666666666|
|      1|      1616|               1.0|
+-------+----------+------------------+
only showing top 5 rows
```

#### Paso 3: Breve análisis descriptivo del dataset.

En este paso se realizan algunas consultas simples sobre la base de datos, combinando la informacion disponible sobre el genero de los usuarios de la plataforma y sus distintas valoraciones. Finalmente se visualiza la distribucion de los datos de ratings para la muestra correspondiente a los 10000 primeros perfiles almacenados en la base de datos, para lo que sera necesario convertir el dataframe de Spark a Pandas.

```python
# Get top 10 most rated profiles (most popular profiles)
top_most_rated_profiles = ratings_df.groupBy("profile_id").count().sort(F.col("count").desc()).limit(50000)
top_most_rated_profiles = top_most_rated_profiles.withColumnRenamed("profile_id", "popular_profile_id")
top_most_rated_profiles.show(10)
------------------------------------------------------------------
|popular_profile_id|count|
+------------------+-----+
|            156148|33389|
|             31116|28398|
|            193687|23649|
|            121859|23639|
|             83773|23113|
|             22319|21387|
|             71636|21284|
|             89855|20634|
|             20737|18550|
|            162707|18224|
+------------------+-----+
only showing top 10 rows
------------------------------------------------------------------
# Get top 10 better rated profiles
avg_rating_by_profile = ratings_df.groupBy("profile_id").agg(F.avg('rating').alias('avg_rating')).sort(F.col("avg_rating").desc())
avg_rating_by_profile.show(10)
------------------------------------------------------------------
+----------+----------+
|profile_id|avg_rating|
+----------+----------+
|    112101|      10.0|
|       898|      10.0|
|    186236|      10.0|
|     36930|      10.0|
|     67625|      10.0|
|     49914|      10.0|
|    159678|      10.0|
|    129824|      10.0|
|     39488|      10.0|
|     66810|      10.0|
+----------+----------+
only showing top 10 rows
------------------------------------------------------------------
# Compute a ration between rating value and popularity
top_profiles = top_most_rated_profiles.join(avg_rating_by_profile, top_most_rated_profiles["popular_profile_id"] == avg_rating_by_profile["profile_id"], "left_outer").drop('profile_id').withColumn("ratio", F.col("avg_rating") / F.col("count"))
# Top profiles sorted by the relation average rating - number of times rated
top_profiles.select("popular_profile_id", "ratio").sort(F.col("ratio").desc()).show(5)
------------------------------------------------------------------
+------------------+------------------+
|popular_profile_id|             ratio|
+------------------+------------------+
|            141167|0.1724137931034483|
|            214508|0.1724137931034483|
|            121533|0.1724137931034483|
|            159206|0.1724137931034483|
|             78319|0.1724137931034483|
+------------------+------------------+
only showing top 5 rows
------------------------------------------------------------------
# Gender of the most popular profile, avoiding Unknown gender profiles
print("ID and gender of the most popular profile:")
top_profiles_gender.filter(gender_df.gender != "U").select("popular_profile_id", "gender", "count", "avg_rating").sort(F.col("count").desc()).show(1)
------------------------------------------------------------------
ID and gender of the most popular profile:
+------------------+------+-----+-----------------+
|popular_profile_id|gender|count|       avg_rating|
+------------------+------+-----+-----------------+
|             31116|     M|28398|7.790583843932671|
+------------------+------+-----+-----------------+
only showing top 1 row
------------------------------------------------------------------
# Gender of the best rated profile
print("ID and gender of the best rated profile:")
top_profiles_gender.filter(gender_df.gender != "U").select("popular_profile_id", "gender", "count", "avg_rating").sort(F.col("avg_rating").desc()).show(1)
------------------------------------------------------------------
ID and gender of the best rated profile:
+------------------+------+-----+----------+
|popular_profile_id|gender|count|avg_rating|
+------------------+------+-----+----------+
|             67169|     M|  102|      10.0|
+------------------+------+-----+----------+
only showing top 1 row
------------------------------------------------------------------
# Average rating by gender
avg_rating_by_gender = top_profiles_gender.filter(gender_df.gender != "U").groupBy("gender").agg(F.avg('avg_rating').alias('avg_rating_by_gender')).sort(F.col("avg_rating_by_gender").desc())
avg_rating_by_gender.show()
------------------------------------------------------------------
+------+--------------------+
|gender|avg_rating_by_gender|
+------+--------------------+
|     M|     6.1910968513945|
|     F|   5.161122966430577|
+------+--------------------+
------------------------------------------------------------------
# Total rates by gender
total_votes_by_gender = top_profiles_gender.filter(gender_df.gender != "U").groupBy("gender").agg(F.sum('count').alias('total_rates_by_gender')).sort(F.col("total_rates_by_gender").desc())
total_votes_by_gender.show()
------------------------------------------------------------------
+------+---------------------+
|gender|total_rates_by_gender|
+------+---------------------+
|     M|              8156451|
|     F|              4509053|
+------+---------------------+
------------------------------------------------------------------
# Convert Spark dataframe to Pandas to plot data distribution
pandas_df = df.limit(10000).toPandas()
# Ratings distribution for a sample of 10000 users
sns.violinplot([pandas_df.label])
```
![_config.yml]({{ site.baseurl }}/images/distribution.png)
