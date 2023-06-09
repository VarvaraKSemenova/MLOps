from pyspark.sql import SparkSession
from pyspark.mllib.random import RandomRDDs
from pyspark.sql.functions import col, monotonically_increasing_id, udf
from pyspark.sql.types import ArrayType, IntegerType, StructType

"""
Generate customers and terminals
"""

n_items_customer = 5  # x_customer_id, y_customer_id, mean_amount, std_amount, mean_nb_tx_per_day
n_items_terminal = 2  # x_terminal_id, y_terminal_id
r_available = 5
spark = SparkSession.builder.appName("generate-clients").getOrCreate()
spark.sparkContext.setLogLevel('WARN')


def generate_customer_profiles_table(n_customers, random_state=None):
    profiles = RandomRDDs.uniformVectorRDD(spark.sparkContext, n_customers, n_items_customer, seed=random_state).map(
        lambda a: a.tolist()).toDF()
    profiles = profiles.withColumnRenamed('_1', 'x_customer_id')
    profiles = profiles.withColumn('x_customer_id', col('x_customer_id') * 100)
    profiles = profiles.withColumnRenamed('_2', 'y_customer_id')
    profiles = profiles.withColumn('y_customer_id', col('y_customer_id') * 100)
    profiles = profiles.withColumnRenamed('_3', 'mean_amount')
    profiles = profiles.withColumn('mean_amount', col('mean_amount') * 95 + 5)
    profiles = profiles.withColumnRenamed('_4', 'mean_nb_tx_per_day')
    profiles = profiles.withColumn('mean_nb_tx_per_day', col('mean_nb_tx_per_day') * 4)
    profiles = profiles.withColumn('CUSTOMER_ID', monotonically_increasing_id())
    profiles = profiles.withColumn('std_amount', col('mean_amount') / 2)
    return profiles


def generate_terminal_profiles_table(n_terminals, random_state=None):
    terminals = RandomRDDs.uniformVectorRDD(spark.sparkContext, n_terminals, n_items_terminal, seed=random_state).map(
        lambda a: a.tolist()).toDF()

    terminals = terminals.withColumnRenamed('_1', 'x_terminal_id')
    terminals = terminals.withColumn('x_terminal_id', col('x_terminal_id') * 100)
    terminals = terminals.withColumnRenamed('_2', 'y_terminal_id')
    terminals = terminals.withColumn('y_terminal_id', col('y_terminal_id') * 100)
    terminals = terminals.withColumn('TERMINAL_ID', monotonically_increasing_id())

    return terminals


terminals_list = generate_terminal_profiles_table(10000)
terminals_list.write.parquet('./terminals_list', mode='overwrite')


customer_list = generate_customer_profiles_table(5000)
collected_dict = {row['TERMINAL_ID']: [row['x_terminal_id'], row['y_terminal_id']] for row in terminals_list.collect()}


def get_available(x, y):
    return list({id_0: coor for id_0, coor in collected_dict.items() if (x - coor[0])*(x - coor[0]) +
                 (y - coor[1])*(y - coor[1]) < r_available*r_available}.keys())


udf_get_available = udf(get_available, ArrayType(IntegerType()))

customer_list = customer_list.withColumn('available_terminals',
                                         udf_get_available(col('x_customer_id'), col('y_customer_id')))

customer_list.write.parquet('customer_list')
