from pyspark.sql import SparkSession, Window
from pyspark.mllib.random import RandomRDDs
from pyspark.sql.types import StructType, TimestampType, IntegerType, StructField, DoubleType
from pyspark.sql.functions import col, from_unixtime, floor, udf, lit, \
    to_timestamp, when, expr, row_number, monotonically_increasing_id
import math
from datetime import datetime
import gc

"""
Generate transactions
"""

n_days = 2000 * 91
start_time = round(datetime(2021, 12, 7).timestamp())
spark = SparkSession.builder.appName("generate-transactions").getOrCreate()
spark.sparkContext.setLogLevel('WARN')
customer_list = spark.read.parquet('/user/dataproc-agent/customer_list')

customer_collect = customer_list.collect()

transactions = spark.createDataFrame(
    [], StructType([
        StructField('TX_DATETIME', TimestampType(), False),
        StructField('CUSTOMER_ID', IntegerType(), False),
        StructField('TERMINAL_ID', IntegerType(), False),
        StructField('TX_AMOUNT', DoubleType(), False),
        StructField('TX_TIME_SECONDS', IntegerType(), False),
        StructField('TX_TIME_DAYS', IntegerType(), False)
    ])
)

transactions.write.csv('transactions_list', mode='overwrite')
counter = 0
n_customer = len(customer_collect)

for row in customer_collect:
    counter += 1
    print(counter, n_customer)
    # Generate amount of transactions for every day
    n_transactions_list = RandomRDDs.poissonRDD(
        spark, row['mean_nb_tx_per_day'], n_days
    ).map(lambda x: math.ceil(x)).map(
        lambda x: [x]
    ).toDF()#.collect()
    terminal_list = row['available_terminals']


    def get_terminal(x):
        return terminal_list[x]
    udf_get_terminal_list = udf(get_terminal, IntegerType())

    # Generate date number for all transactions
    days_labels_list = n_transactions_list.withColumn(
        'id_n', monotonically_increasing_id()
    ).withColumn(
        'TX_TIME_DAYS', row_number().over(window=Window.orderBy('id_n'))
    ).drop('id_n').withColumnRenamed(
        '_1', 'n'
    ).withColumn(
        'n', expr('explode(array_repeat(n,int(n)))')
    ).drop('n').withColumn(
        'id_n', monotonically_increasing_id()
    ).withColumn('id', row_number().over(window=Window.orderBy('id_n'))).drop('id_n')

    # Get total amount of transactions
    n_gen = days_labels_list.count()
    print('\nn_gen = ', n_gen)
    if n_gen == 0:
        continue

    # Generate terminal_id number for all transactions
    df_terminal_id = RandomRDDs.uniformRDD(spark, n_gen).map(lambda x: [x]).toDF()
    weight = 1 / len(terminal_list)
    df_terminal_id = df_terminal_id.withColumn(
        '_1', floor(col('_1') / weight)
    )
    df_terminal_id = df_terminal_id.withColumnRenamed('_1', 'TERMINAL_ID')
    df_terminal_id = df_terminal_id.withColumn(
        'id_n', monotonically_increasing_id()
    ).withColumn('id', row_number().over(window=Window.orderBy('id_n'))).drop('id_n')

    # Generate amount for all transactions
    df_tx_amount = RandomRDDs.normalRDD(spark, n_gen).map(
        lambda x: row['mean_amount'] + row['std_amount'] * x
    ).map(
        lambda x: [x]
    ).toDF()
    df_tx_amount = df_tx_amount.withColumnRenamed('_1', 'TX_AMOUNT')
    df_tx_amount = df_tx_amount.withColumn(
        'id_n', monotonically_increasing_id()
    ).withColumn('id', row_number().over(window=Window.orderBy('id_n'))).drop('id_n')

    tmp = RandomRDDs.uniformRDD(
        spark, n_gen
    ).map(
        lambda x: [x * row['mean_amount'] * 2]
    ).toDF().withColumnRenamed('_1', 'x').withColumn(
        'id_n', monotonically_increasing_id()
    ).withColumn('id', row_number().over(window=Window.orderBy('id_n'))).drop('id_n')

    df_tx_amount = df_tx_amount.join(
        tmp, on='id'
    ).withColumn(
        'TX_AMOUNT', when(col('TX_AMOUNT') > 0, col('TX_AMOUNT')).otherwise(col('x'))
    ).drop('x')

    # Generate timepoints for all transactions
    df_tx_timeseconds = RandomRDDs.normalRDD(spark, n_gen).map(
        lambda x: round(86400 / 2 + 20000 * x)
    ).map(
        lambda x: [x]
    ).toDF()
    df_tx_timeseconds = df_tx_timeseconds.withColumnRenamed('_1', 'TX_TIME_SECONDS')
    df_tx_timeseconds = df_tx_timeseconds.withColumn(
        'id_n', monotonically_increasing_id()
    ).withColumn('id', row_number().over(window=Window.orderBy('id_n'))).drop('id_n')

    # Join all information
    df_tx_timeseconds = df_tx_timeseconds.join(
        df_terminal_id, on=['id']
    ).join(
        df_tx_amount, on=['id']
    ).join(
        days_labels_list, on=['id']
    ).filter(
        col('TX_TIME_SECONDS') > 0
    ).filter(
        col('TX_TIME_SECONDS') < 86400
    ).drop('id').withColumn(
        'CUSTOMER_ID', lit(row['CUSTOMER_ID'])
    ).withColumn(
        'TX_DATETIME', col('TX_TIME_SECONDS') + col('TX_TIME_DAYS') * 86400 + start_time
    ).withColumn(
        'TX_DATETIME', to_timestamp(from_unixtime(col('TX_DATETIME')))
    )

    df_tx_timeseconds = df_tx_timeseconds.select('TX_DATETIME', 'CUSTOMER_ID',
                                                 'TERMINAL_ID',
                                                 'TX_AMOUNT', 'TX_TIME_SECONDS',
                                                 'TX_TIME_DAYS')

    # Get actual values of terminal id
    df_tx_timeseconds = df_tx_timeseconds.withColumn('TERMINAL_ID', udf_get_terminal_list(col('TERMINAL_ID')))
    df_tx_timeseconds.write.csv('transactions_list', mode='append')

    gc.collect()

    df_tx_timeseconds.unpersist()
    df_terminal_id.unpersist()
    df_tx_amount.unpersist()
    n_transactions_list.unpersist()
    days_labels_list.unpersist()
