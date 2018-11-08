from pyspark.sql import SparkSession, Row
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS

INP_PATH = '../bin/c&p_data_all.json'
OUT_PATH = '../bin/c&p_data_5000.json'
# the number of recommended products
K = 5


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
    print('After the conversion:')
    res.show()
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
    training, test = df.randomSplit([0.8, 0.2])

    print("++++++++++++++++%d" % training.select('cid').distinct().count())

    alsExplicit = ALS(maxIter=5, regParam=0.01, userCol="cid", itemCol="pid", ratingCol="score")
    modelExplicit = alsExplicit.fit(training)
    predictionsExplicit = modelExplicit.transform(test)
    predictionsExplicit.orderBy('cid').show()

    userRecs = modelExplicit.recommendForAllUsers(10)
    userRecs.show()



if __name__ == '__main__':

    spark = SparkSession.builder.getOrCreate()

    # extract_top_N(5000)

    df = spark.read.json(OUT_PATH)
    df = numerical_converter(df)



    collaborative_filter(df)





