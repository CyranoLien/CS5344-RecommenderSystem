def temp():
    from pyspark.sql import SparkSession
    from pyspark.ml.fpm import FPGrowth
    spark = SparkSession.builder.getOrCreate()

    df = spark.createDataFrame([
        (0, ['a', 'b', 'e']),
        (1, ['a', 'b', 'c', 'e']),
        (2, ['a', 'b'])
    ], ["id", "items"])

    fpGrowth = FPGrowth(itemsCol="items", minSupport=0.5, minConfidence=0.6)
    model = fpGrowth.fit(df)

    # Display frequent itemsets.
    model.freqItemsets.show()

    # Display generated association rules.
    model.associationRules.show()

    # transform examines the input items against all the association rules and summarize the
    # consequents as prediction
    model.transform(df).show()

a = [1,2,3,4]
b =[4]


x = set(a)
y = set(b)

def s(x,y):
    return 0 if len(x&y) is 0 else 1

print(s(x,y))