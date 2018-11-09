from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as F
from pyspark.sql.types import *

INP_PATH = '../bin/c&p_data_all.json'
OUT_PATH = '../bin/c&p_data_5000.json'
# the number of recommended products
K = 10


def extract_top_N(n):
    with open(INP_PATH, 'r') as f1:
        with open(OUT_PATH, 'w') as f2:
            i = 0
            for line in f1.readlines():
                f2.write(line)
                if i < n:
                    i += 1
                else:
                    break


def numerical_converter(df):
    # before the conversion
    # print('Before the conversion:')
    # df.show()

    # convert customerID into numerical index
    indexer1 = StringIndexer(inputCol="customerID", outputCol="cid")
    model1 = indexer1.fit(df)
    # convert productID into numerical index
    indexer2 = StringIndexer(inputCol="productID", outputCol="pid")
    model2 = indexer2.fit(df)
    indexed = model2.transform(model1.transform(df))

    # after the conversion
    res = indexed.select(indexed.cid, indexed.pid, indexed.score)
    # print('After the conversion:')
    # res.show()
    return res


def select_min_5(df):
    df.createOrReplaceTempView('table0')

    counts = df.groupby('pid').count()
    counts = counts.filter('count > %d' % 5)
    counts.show()
    counts.createOrReplaceTempView('valid_pid')

    res = spark.sql("SELECT * FROM table0 WHERE table0.pid IN (SELECT pid FROM valid_pid)")
    res.orderBy('pid', ascending=False)
    return res


def collaborative_filter(df):
    # TODO: sample testset from those whose scores are high
    # _df = df.orderBy(F.rand()).orderBy('score', ascending=False)
    # print(_df.count())
    # test = _df.limit(50000).orderBy('pid')
    # print(test.count())
    # training = _df.subtract(test).orderBy('pid')
    # print(training.count())

    training, test = df.randomSplit([0.9, 0.1])

    # TODO: train the model
    alsExplicit = ALS(maxIter=5, regParam=0.01, userCol="cid", itemCol="pid", ratingCol="score")
    modelExplicit = alsExplicit.fit(training)

    # TODO: test the model
    # pre-process the test data
    res_truth = test.groupby('cid').agg(F.collect_list('pid').alias('ground_truth')).orderBy('cid')

    # TODO: generate the prediction for test data
    user = test.select('cid').distinct()
    res_pre = modelExplicit.recommendForUserSubset(user, K).orderBy('cid')
    # define an UDF to transform the prediction output
    my_udf = lambda x: [i[0] for i in x]
    extract = F.udf(my_udf, ArrayType(IntegerType()))
    res_pre = res_pre.withColumn('prediction', extract(res_pre.recommendations))
    res = res_truth.join(res_pre, res_truth.cid == res_pre.cid).select(res_truth["*"], res_pre["prediction"])
    # calculate the conversion rate
    conversion = F.udf(lambda x, y: 0 if len(set(x) & set(y)) == 0 else 1, IntegerType())
    res = res.withColumn('conversion', conversion('ground_truth', 'prediction'))

    res.show()
    print("The total number of transaction in testset is: %d     K = %d" % (test.count(), K))
    print("The number of total users in the testset is: %d" % res.count())
    total_v = res.agg(F.sum('conversion')).collect()[0][0]
    print("The number of converted users is: %d" % total_v)


    '''
    predictionsExplicit = modelExplicit.transform(test)
    predictionsExplicit.orderBy('cid').show()
    '''


if __name__ == '__main__':

    spark = SparkSession.builder.getOrCreate()

    # extract_top_N(5000)

    df = spark.read.json(INP_PATH)
    df = numerical_converter(df)
    collaborative_filter(df)





