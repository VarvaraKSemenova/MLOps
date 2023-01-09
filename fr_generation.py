from pyspark.sql import SparkSession, Window
from pyspark.mllib.random import RandomRDDs
from pyspark.sql.functions import col, from_unixtime, floor, udf, lit, \
    to_timestamp, when, expr, row_number, monotonically_increasing_id, array, explode
from pyspark.sql.types import StructType, TimestampType, IntegerType, StructField, DoubleType
# Left join
# TX_FRAUD, TX_FRAUD_SCENARIO
spark = SparkSession.builder.appName("generate-transactions").getOrCreate()
# spark.sparkContext.setLogLevel('WARN')
customer_list = spark.read.parquet('/user/dataproc-agent/customer_list')
terminals_list = spark.read.parquet('/user/dataproc-agent/terminals_list')
transaction_list = spark.read.csv('/user/dataproc-agent/transactions_list')
#
# 'TX_DATETIME', 'CUSTOMER_ID',
#                                                  'TERMINAL_ID',
#                                                  'TX_AMOUNT', 'TX_TIME_SECONDS',
#                                                  'TX_TIME_DAYS'
transaction_list = transaction_list.withColumnRenamed('_c0', 'TX_DATETIME')
transaction_list = transaction_list.withColumnRenamed('_c1', 'CUSTOMER_ID')
transaction_list = transaction_list.withColumnRenamed('_c2', 'TERMINAL_ID')
transaction_list = transaction_list.withColumnRenamed('_c3', 'TX_AMOUNT')
transaction_list = transaction_list.withColumnRenamed('_c4', 'TX_TIME_SECONDS')
transaction_list = transaction_list.withColumnRenamed('_c5', 'TX_TIME_DAYS')
transaction_list.show()
n_cump_term_days = 28
n_cump_cust_days = 14

customer_list = [row['CUSTOMER_ID'] for row in customer_list.select('CUSTOMER_ID').collect()]
weight_customer = len(customer_list)
terminals_list = [row['TERMINAL_ID'] for row in terminals_list.select('TERMINAL_ID').collect()]
weight_terminals = len(terminals_list)


def get_terminal(x):
    return terminals_list[x]


udf_get_terminal = udf(get_terminal, IntegerType())


def get_customer(x):
    return customer_list[x]


udf_get_customer = udf(get_customer, IntegerType())

print('\nGet max')
n_days = 2000 * 91 # transaction_list.select('TX_TIME_DAYS').rdd.max()[0]
transaction_list = transaction_list.filter(col('TX_TIME_DAYS') <= 2000*20*4).filter(col('TX_TIME_DAYS') > 2000*20*3)
print('Max = ', n_days)
n_compr_terminals = 2
n_compr_customers = 2

# print('\nGenerate ter')
# compromised_terminals_list = RandomRDDs.uniformVectorRDD(
#     spark.sparkContext, n_days, n_compr_terminals
# ).map(
#         lambda a: a.tolist()
# ).toDF()
# print('\nTransform ter')
# compromised_terminals_list = compromised_terminals_list.withColumn(
#     '_1', floor(col('_1') / weight_terminals)
# ).withColumn(
#     '_2', floor(col('_2') / weight_terminals)
# ).withColumnRenamed('_1', 'ter_1').withColumnRenamed('_2', 'ter_2').withColumn(
#     'ter_2', when(col('ter_1') == col('ter_2'), col('ter_2')+lit(1)).otherwise(col('ter_2'))
# ).withColumn(
#     'ter_1', udf_get_terminal(col('ter_1'))
# ).withColumn(
#     'ter_2', udf_get_terminal(col('ter_2'))
# ).withColumn(
#     'id_n', monotonically_increasing_id()
# ).withColumn('day', row_number().over(window=Window.orderBy('id_n'))).drop('id_n')
#
# compromised_terminals_list.write.csv('compromised_terminals', mode='overwrite', header='True')
compromised_terminals_list = spark.read.csv('/user/dataproc-agent/compromised_terminals',  header='True')

compromised_terminals_list = compromised_terminals_list.withColumn(
    'term', array(col('ter_1'), col('ter_2'))
).drop('ter_1').drop('ter_2').withColumn(
    'term', explode(col('term'))
).withColumn('compr_term', lit(1)).withColumn(
    'ndays', array([lit(x) for x in range(n_cump_term_days)])
).withColumn(
    'ndays', explode(col('ndays'))
).withColumn('day', col('day') + col('ndays')).drop('ndays')

# print('\nGenerate cus')
# compromised_customers_list = RandomRDDs.uniformVectorRDD(
#     spark.sparkContext, n_days, n_compr_customers
# ).map(
#         lambda a: a.tolist()
# ).toDF()
#
#
# print('\nTransform cus')
# compromised_customers_list = compromised_customers_list.withColumn(
#     '_1', floor(col('_1') / weight_customer)
# ).withColumn(
#     '_2', floor(col('_2') / weight_customer)
# ).withColumnRenamed('_1', 'cust_1').withColumnRenamed('_2', 'cust_2').withColumn(
#     'cust_2', when(col('cust_1') == col('cust_2'), col('cust_2')+lit(1)).otherwise(col('cust_2'))
# ).withColumn(
#     'cust_1', udf_get_terminal(col('cust_1'))
# ).withColumn(
#     'cust_2', udf_get_terminal(col('cust_2'))
# ).withColumn(
#     'id_n', monotonically_increasing_id()
# ).withColumn('day', row_number().over(window=Window.orderBy('id_n'))).drop('id_n')
#
# compromised_customers_list.write.csv('compromised_customers', mode='overwrite', header='True')

compromised_customers_list = spark.read.csv('/user/dataproc-agent/compromised_customers',  header='True')

compromised_customers_list = compromised_customers_list.withColumn(
    'cust', array(col('cust_1'), col('cust_2'))
).drop('cust_1').drop('cust_2').withColumn(
    'cust', explode(col('cust'))
).withColumn('compr_cust', lit(1)).withColumn(
    'ndays', array([lit(x) for x in range(n_cump_cust_days)])
).withColumn(
    'ndays', explode(col('ndays'))
).withColumn('day', col('day') + col('ndays')).drop('ndays')

print('\nFirst joint started!')
transaction_list = transaction_list.join(
    compromised_terminals_list, (
        (transaction_list['TERMINAL_ID'] == compromised_terminals_list['term']) &
        (transaction_list['TX_TIME_DAYS'] == compromised_terminals_list['day'])
    ), 'left'
).drop('day', 'term')
print('\nSecond join started!')
transaction_list = transaction_list.join(
    compromised_customers_list, (
        (transaction_list['CUSTOMER_ID'] == compromised_customers_list['cust']) &
        (transaction_list['TX_TIME_DAYS'] == compromised_customers_list['day'])
    ), 'left'
).drop('day', 'cust')
print('\nSecond join finished')

transaction_list = transaction_list.fillna(0)

print('\nset TX_FRAUD_SCENARIO')
transaction_list = transaction_list.withColumn(
    'TX_FRAUD_SCENARIO',
    when(col('compr_cust') > 0, lit(3)).when(
        col('compr_term') > 0, lit(2)).when(col('TX_AMOUNT') > 220, lit(1)).otherwise(lit(0))
)
print('\nset TX_FRAUD')
transaction_list = transaction_list.withColumn(
    'TX_FRAUD', when(col('TX_FRAUD_SCENARIO') > 0, lit(1)).otherwise(lit(0))
)
print('\nset TX_AMOUNT')
transaction_list = transaction_list.withColumn(
    'TX_AMOUNT',
    when(col('TX_FRAUD_SCENARIO') == 3, col('TX_AMOUNT') * lit(5)).otherwise(col('TX_AMOUNT'))
).drop('compr_cust', 'compr_term')

compromised_terminals_list.unpersist()
compromised_customers_list.unpersist()
print('start write')
# transaction_list.show()

transaction_list.write.csv('transactions_list_with_marks', mode='append', header='True')
