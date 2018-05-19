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

#### Paso 4: Selección del modelo y ajuste de hiperparámetros.

El proceso de selección del modelo se realizará a través del análisis de validación cruzada con ajuste automático de hiperparámetros. Este ajuste se hace definiendo los posibles valores de los hiperparámetros del modelo y ejecutando una búsqueda en rejilla sobre éstas para comparar el rendimiento de los modelos resultantes y finalmente obtener el óptimo. Los hiperparámetros del modelo ALS son:
rank = la cantidad de factores latentes en el modelo (4, 8 y 12 como valores seleccionados)
maxIter = el número máximo de iteraciones (valor predeterminado)
regParam = el parámetro de regularización (0.1, 0.05 y 0.01 como valores seleccionados)

El proceso anterior puede implementarse rápidamente utilizando la función Spark CrossValidator.
Para realizar una comparativa entre los modelos obtenidos con el proceso anterior se establece como método de evaluación el cálculo del Error cuadrático medio (RMSE) ya que se usa comúnmente como principal métrica de evaluación en problemas de regresión y está disponible en la API de Spark. RMSE compara los valores predichos del conjunto de entrenamiento con los valores reales presentes en el conjunto de validación, al agregar el error absoluto de las diferencias y tomar el promedio de esos valores obtenemos una medida del error del modelo. Cuanto menor es el error, mejor es la capacidad de pronóstico de ese modelo según el criterio RMSE. También se obtendrán otras métricas del error como MSE, R2 y MAE.
Sin embargo, los libros de texto y artículos de investigación en el campo (por ejemplo, F.O. Isinkayea, Y.O. Folajimib, B.A. Ojokohc, 2015) recomiendan usar evaluaciones similares a las de RankMetrics para calcular métricas como la precisión promedio en K o MAP.

```python
# Split into train and test subsets
(training, test) = df.randomSplit([0.8, 0.2])
cv_data = training.withColumnRenamed("norm_rating", "label")
# Set model
als = ALS(userCol="user_id", itemCol="profile_id", ratingCol="label", coldStartStrategy="drop", seed=0, nonnegative=True)
# Set considered parameter grid
paramGrid = ParamGridBuilder().addGrid(als.regParam, [0.1, 0.05, 0.01]).addGrid(als.rank, [4, 8, 12]).build()
# Set evaluator
modelEvaluator = RegressionEvaluator(metricName="rmse")
# Set cross validator instance
crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=paramGrid,
                          evaluator=modelEvaluator,
                          numFolds=10)
# Perform cross-validation
cvModel = crossval.fit(cv_data)
# Select best model and get its parameters
best_als_model = cvModel.bestModel
print("Best number of latent factors (rank parameter): " + str(best_als_model.rank))
print("Best value of regularization factor: " + str(best_als_model._java_obj.parent().getRegParam()))
print("Max Iterations: " + str(best_als_model._java_obj.parent().getMaxIter()))
------------------------------------------------------------------
Best number of latent factors (rank parameter): 8
Best value of regularization factor: 0.01
Max Iterations: 10
------------------------------------------------------------------
# Make predictions on a random test subset obtained through randomSplit()
print("Predictions based on a random test subset:")
predictions = best_als_model.transform(test)
predictions.show(5)
# Evaluate model's performance on test (evaluate overfitting)
def overfitting_evaluation(predictions):
    # Model evaluation in test - ratings regression evaluation
    print("Model evaluation on test data:")
    predictions = predictions.na.drop()
    # RMSE
    rmse_evaluator = RegressionEvaluator(metricName="rmse", labelCol="label", predictionCol="prediction")
    rmse = rmse_evaluator.evaluate(predictions)
    print("Root-mean-square error (RMSE) = " + str(rmse))
    # MSE
    mse_evaluator = RegressionEvaluator(metricName="mse", labelCol="label", predictionCol="prediction")
    mse = mse_evaluator.evaluate(predictions)
    print("Mean-square error (MSE) = " + str(mse))
    # R2
    r2_evaluator = RegressionEvaluator(metricName="r2", labelCol="label", predictionCol="prediction")
    r2 = r2_evaluator.evaluate(predictions)
    print("r² metric = " + str(r2))
    # MAE
    mae_evaluator = RegressionEvaluator(metricName="mae", labelCol="label", predictionCol="prediction")
    mae = mae_evaluator.evaluate(predictions)
    print("Mean Absolute Error (MAE) = " + str(mae))
return [rmse, mse, r2, mae]
random_test_eval = overfitting_evaluation(predictions)
------------------------------------------------------------------
Predictions based on a random test subset:
+-------+----------+------------------+----------+
|user_id|profile_id|             label|prediction|
+-------+----------+------------------+----------+
|  83775|       496|0.6666666666666666| 0.5487299|
|  83524|       496|               1.0| 0.9750136|
|  28584|       833|               1.0| 0.7296401|
| 114979|      1238|0.6666666666666666| 0.5668328|
|  96625|      1238|0.6666666666666666|0.43366265|
+-------+----------+------------------+----------+
only showing top 5 rows

Model evaluation on test data:
Root-mean-square error (RMSE) = 0.204187073287
Mean-square error (MSE) = 0.0416923608976
r² metric = 0.650955563635
Mean Absolute Error (MAE) = 0.140817836752
```
Para obtener una evaluación más precisa del sobreajuste del modelo y del rendimiento real, una buena práctica es, una vez que se ha seleccionado el modelo ALS con el mejor ajuste de hiperparámetros (proceso de selección del modelo, realizado por validación cruzada con búsqueda en rejilla), se evalua el modelo ajustado para distintos conjuntos de entrenamiento y test, seleccionados aleatoriamente a través de múltiples K-fold y, finalmente, se promedian los resultados de las distintas métricas de evaluación calculadas en cada K-fold. Este proceso nos proporcionará una evaluación más precisa de nuestro motor de recomendación como predictor o regresor de ratings para nuevos datos entrantes nunca antes vistos por el sistema.

```python
def kfold_test_eval(df, Kfolds=5):
    rmse_evaluations = []
    mse_evaluations = []
    r2_evaluations = []
    mae_evaluations = []
    
    for k in range(0, Kfolds):  
        (train, test) = df.randomSplit([0.8, 0.2])
        tunned_als = als = ALS(userCol="user_id", itemCol="profile_id", ratingCol="label", coldStartStrategy="drop", maxIter=10, regParam=0.01, rank=8)
        model = tunned_als.fit(train)
        predictions = model.transform(test)
        print("Kfold: " + str(k + 1))
        k_test_eval = overfitting_evaluation(predictions)
        rmse_evaluations.append(k_test_eval[0])
        mse_evaluations.append(k_test_eval[1])
        r2_evaluations.append(k_test_eval[2])
        mae_evaluations.append(k_test_eval[3])
        
    average_rmse = sum(rmse_evaluations)/float(len(rmse_evaluations))
    average_mse = sum(mse_evaluations)/float(len(mse_evaluations))
    average_r2 = sum(r2_evaluations)/float(len(r2_evaluations))
    average_mae = sum(mae_evaluations)/float(len(mae_evaluations))
    
    return [average_rmse, average_mse, average_r2, average_mae]
[average_rmse, average_mse, average_r2, average_mae] = kfold_test_eval(df)
# Average performance score of the selected model:
print("Average Root-mean-square error (RMSE) = " + str(average_rmse))
print("Average Mean-square error (MSE) = " + str(average_mse))
print("Average r² metric = " + str(average_r2))
print("Average Mean Absolute Error (MAE) " + str(average_mae))
------------------------------------------------------------------
Average Root-mean-square error (RMSE) = 0.203900591499
Average Mean-square error (MSE) = 0.0415754551043
Average r² metric = 0.652060716195
Average Mean Absolute Error (MAE) 0.140639479009
```

#### Paso 5: Comprobar las predicciones de ratings que se obtienen del modelo final para distintos casos.

```python
# Generate top 10 profiles recommendations for each user
userRecs = ratings_predictor.recommendForAllUsers(10)
userRecs.show(1, truncate=False)
+-------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|user_id|recommendations                                                                                                                                                                                             |
+-------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|148    |[[159513, 2.6826572], [135056, 2.2142038], [80297, 2.181056], [199770, 2.0491154], [67442, 1.9784403], [22972, 1.9325198], [128646, 1.9314922], [56050, 1.9247315], [96515, 1.8710849], [179283, 1.8512474]]|
+-------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
only showing top 1 row
# Generate top 10 user recommendations for each profile
profileRecs = ratings_predictor.recommendForAllItems(10)
profileRecs.show(1, truncate=False)
+----------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|profile_id|recommendations                                                                                                                                                                                         |
+----------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|496       |[[10115, 1.7967153], [36243, 1.6167616], [123794, 1.5479522], [79671, 1.5467885], [38123, 1.5259393], [65649, 1.5140268], [66232, 1.4930097], [20044, 1.4844044], [26508, 1.483938], [61468, 1.4789674]]|
+----------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
only showing top 1 row
# Generate top 10 profile recommendations for a set of 10 users
users = test.select(als.getUserCol()).distinct().limit(10)
userSubsetRecs = ratings_predictor.recommendForUserSubset(users, 10)
userSubsetRecs.show(truncate=False)
+-------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|user_id|recommendations                                                                                                                                                                                                |
+-------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|1580   |[[73450, 2.1160717], [196118, 1.9395694], [44768, 1.861454], [46427, 1.8534217], [62877, 1.800837], [121938, 1.7637302], [35786, 1.7616118], [218717, 1.7597885], [185904, 1.7546182], [114965, 1.7449896]]    |
|471    |[[56050, 1.8613569], [169405, 1.6265004], [141912, 1.6238357], [199020, 1.5705091], [153801, 1.5702825], [37519, 1.5582786], [168597, 1.5427593], [117322, 1.5213909], [200918, 1.4999101], [90154, 1.4913626]]|
|1591   |[[169405, 1.6842599], [216493, 1.6176178], [7782, 1.5960064], [37519, 1.5920119], [168597, 1.5863789], [56050, 1.5832279], [199020, 1.5650901], [107089, 1.5618131], [200918, 1.5434403], [111214, 1.4910805]] |
|1342   |[[49049, 1.8087612], [118858, 1.7899785], [219453, 1.7231553], [117973, 1.7054595], [178022, 1.6659847], [143675, 1.652686], [140886, 1.6521295], [159317, 1.6423284], [203180, 1.6299864], [96216, 1.6189286]]|
|463    |[[73450, 2.677851], [162209, 1.9098501], [196962, 1.9062479], [179505, 1.8220359], [197634, 1.8119749], [17567, 1.7738258], [219453, 1.7721505], [105422, 1.761188], [47023, 1.7570614], [23472, 1.7502776]]   |
|833    |[[80297, 1.6658962], [178535, 1.6586927], [111214, 1.4727457], [21216, 1.4659634], [167313, 1.4581232], [135056, 1.4345964], [56050, 1.4294059], [169405, 1.4051217], [78721, 1.4015284], [49602, 1.395864]]   |
|496    |[[167826, 2.0511458], [209453, 1.9527715], [143675, 1.9430621], [205585, 1.939534], [73450, 1.9388179], [219453, 1.9369094], [185904, 1.9364448], [19635, 1.9134604], [141762, 1.8912288], [140446, 1.8898547]]|
|148    |[[159513, 2.6826572], [135056, 2.2142038], [80297, 2.181056], [199770, 2.0491154], [67442, 1.9784403], [22972, 1.9325198], [128646, 1.9314922], [56050, 1.9247315], [96515, 1.8710849], [179283, 1.8512474]]   |
|1088   |[[190926, 2.1182022], [159513, 2.0326533], [76186, 1.9888866], [80297, 1.954988], [18218, 1.901816], [21662, 1.8908844], [199770, 1.8749667], [155396, 1.8639369], [84832, 1.8371751], [15377, 1.8299328]]     |
|1238   |[[127528, 2.096571], [24639, 1.916371], [169405, 1.8475499], [117787, 1.7302287], [168597, 1.7244738], [144968, 1.6855974], [208922, 1.6755896], [187910, 1.6324193], [213142, 1.619712], [197550, 1.619522]]  |
+-------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# Generate top 10 user recommendations for a set of 10 profiles
profiles = test.select(als.getItemCol()).distinct().limit(10)
profileSubSetRecs = ratings_predictor.recommendForItemSubset(profiles, 10)
profileSubSetRecs.show(truncate=False)
+----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|profile_id|recommendations                                                                                                                                                                                                  |
+----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|128131    |[[88607, 1.2465376], [40188, 1.2444562], [69236, 1.2178707], [82086, 1.2172562], [42639, 1.2150319], [121684, 1.2045647], [91642, 1.1989311], [10115, 1.1987841], [29150, 1.1812139], [66232, 1.1797979]]        |
|87462     |[[26871, 1.207882], [81073, 1.2014818], [83148, 1.1523317], [29112, 1.136605], [91642, 1.1039877], [26544, 1.0997831], [118342, 1.0979117], [51208, 1.0882537], [80161, 1.0850104], [121846, 1.0848624]]         |
|205392    |[[112652, 1.241207], [29150, 1.228986], [59406, 1.1866542], [79155, 1.1800299], [29996, 1.1729485], [121684, 1.1671597], [1384, 1.1544853], [87323, 1.1533304], [53778, 1.1521593], [25233, 1.14995]]            |
|203894    |[[10115, 0.79588294], [36243, 0.7817952], [121684, 0.73701406], [38123, 0.7342589], [88607, 0.73281974], [66232, 0.72805846], [123794, 0.7247406], [34941, 0.72130567], [90661, 0.7179232], [42247, 0.7051606]]  |
|44437     |[[101074, 0.68464667], [89420, 0.67187476], [134780, 0.6624583], [29112, 0.6527287], [57924, 0.6328629], [90124, 0.62214154], [90694, 0.6128592], [56006, 0.6121998], [44795, 0.61183506], [79386, 0.609617]]    |
|45307     |[[96042, 0.8553329], [93775, 0.84709346], [28601, 0.84427464], [61468, 0.83761144], [67624, 0.8250998], [92606, 0.81036466], [34956, 0.80686474], [18068, 0.80557084], [118820, 0.799636], [53215, 0.7982516]]   |
|58797     |[[106893, 1.7288352], [93775, 1.6686077], [49177, 1.6475681], [117178, 1.6382982], [29377, 1.5988771], [44795, 1.5766393], [28601, 1.5587692], [66989, 1.5518527], [79468, 1.5421674], [71088, 1.5284784]]       |
|41988     |[[106893, 1.1461915], [96042, 1.1183972], [47540, 1.1158137], [93775, 1.1020718], [95251, 1.0925175], [59908, 1.0858284], [32548, 1.080138], [61468, 1.0478147], [67624, 1.0465776], [28601, 1.0360357]]         |
|104508    |[[25233, 0.86325073], [29150, 0.86161196], [15520, 0.82333964], [92621, 0.8047507], [14712, 0.7958485], [72021, 0.77550006], [89329, 0.7706092], [50241, 0.7615911], [51565, 0.7560502], [14114, 0.7536515]]     |
|186039    |[[66218, 0.25599483], [128641, 0.25121227], [13264, 0.23711407], [92621, 0.2368906], [12228, 0.2360465], [67624, 0.23511758], [133021, 0.23419699], [115540, 0.23253503], [5568, 0.23180199], [7188, 0.23102544]]|
+----------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
# Generate recommendations for a specific user
def profiles4userID(als_model, uid, limit=10):
    data = df.select("profile_id").distinct().withColumn("user_id", F.lit(uid))
    rated_profiles = df.filter(df.user_id == uid).select("profile_id", "user_id")
    predictions = als_model.transform(data.subtract(rated_profiles)).dropna().orderBy("prediction", ascending=False).limit(limit).select("profile_id", "prediction")
    predictions.join(gender_df.filter(gender_df.gender != "U"), predictions.profile_id == gender_df.profile_id).select(predictions.profile_id, gender_df.gender, predictions.prediction).show(1)
gender_df.filter(gender_df.profile_id == "1000").show()
print("Recommended contact for user 1000:")
profiles4userID(best_als_model, 1000)
------------------------------------------------------------------
+----------+------+
|profile_id|gender|
+----------+------+
|      1000|     M|
+----------+------+

Recommended contact for user 1000:
+----------+------+----------+
|profile_id|gender|prediction|
+----------+------+----------+
|     13308|     F| 1.4781343|
+----------+------+----------+
only showing top 1 row
```

#### Conclusiones

Finalmente, hay que señalar que las predicciones obtenidas a través del algoritmo ALS con retroalimentación implícita disponible en PySpark no son valores normalizados, existiendo incluso la posibilidad de forzar a valores no negativos mediante el argumento del constructor de la clase ALS, nonnegative. Asimismo pueden aparecer valores  NaN tras ejecutar el proceso de validación cruzada, a través de la clase de PySpark CrossValidator, haciendo uso de una métrica de evaluación de regresión (RMSE, MSE, R2, MAE), en el caso de que el subconjunto de datos de validación/test contenga usuarios/items no presentes en el subconjunto de datos de entrenamiento, las soluciones se proponen en SPARK-14489 y SPARK-19345.
Apache Flink también ofrece una implementación del algoritmo ALS para filtrado colaborativo, cuya ejecución haciendo uso de éste y otros datasets benchmark sería interesante comparar con la realizada aquí en PySpark, disponible en mi github: https://github.com/aracelimanzanoch/simple-pyspark-recommedation-engine/blob/master/simple-pyspark-recommendation-engine.ipynb

Aunque el sistema en producción requeriría mayor ajuste, complejidad, versatilidad y eficiencia, debiendo actualizar las predicciones en streaming (solución asimismo implementable con Spark o Flink), espero que este primer acercamiento al Filtrado Colaborativo haya sido de utilidad.

See y'all on my social networks!
