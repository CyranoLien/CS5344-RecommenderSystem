from itertools import chain
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window


INP_PATH = '../bin/only_p_all.json'
OUT_PATH = '../bin/only_p_5000.json'

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


def concat(type):
    def concat_(*args):
        return list(chain.from_iterable((arg if arg else [] for arg in args)))
    return F.udf(concat_, ArrayType(type))


def data_process(df):
    train_temp, test_temp = df.randomSplit([0.85, 0.15])

    # for training set, concatenate 'item' with 'also_bought'
    concat_string_arrays = concat(StringType())
    # train_temp.select(concat_string_arrays('item', 'also_bought')).show(truncate=False)
    training = train_temp.withColumn('itemset', concat_string_arrays('item', 'also_bought'))

    # for test set, rename 'item' by 'itemset', 'also_bought' by 'ground_truth'
    test = test_temp.selectExpr("item as itemset", "also_bought as ground_truth")

    return training, test


def get_valid_rules(df):
    rules = df.where(F.size(F.col('antecedent')) == 1)
    rules.createOrReplaceTempView('rules')

    # For each antecedent, order and filter the top k transaction
    # SELECT * FROM rules ORDER BY confidence DESC LIMIT 5 '[B00A4KVFLY]'
    ranked = rules.withColumn("rank", F.row_number().over(Window.partitionBy("antecedent").orderBy(F.desc("confidence"))))
    ranked = ranked.filter(ranked.rank <= K)

    # Very important! Have to convert consequent (ArrayType) into product (StringType)
    # Otherwise in following collect_list the element in the gerenated list is in ArrayTYpe
    res = ranked.withColumn('product', F.concat_ws(',', 'consequent'))
    res = res.groupby('antecedent').agg(F.collect_list('product').alias('prediction'))
    return res


def fp_growth(df):
    training, test = data_process(df)
    fpGrowth = FPGrowth(itemsCol='itemset', minSupport=0.1, minConfidence=0.2)
    model = fpGrowth.fit(training)

    # Display frequent itemsets.
    # model.freqItemsets.show()

    # Display generated association rules.
    rules = model.associationRules
    rules = get_valid_rules(rules)
    rules.show()

    # Display the predicted purchasing.
    # res = model.transform(test).orderBy('prediction', ascending=False)

    # Calculate conversion rate.
    res = test.join(rules, test.itemset == rules.antecedent).select(test["*"], rules["prediction"])
    conversion = F.udf(lambda x, y: 0 if len(set(x) & set(y)) == 0 else 1, IntegerType())
    res = res.withColumn('conversion', conversion('ground_truth', 'prediction'))
    res.show()

    print("The total size of testset is: %d     K = %d" % (test.count(), K))
    total_c = res.count()
    print("The number of total recommendation is: %d" % total_c)
    total_v = res.agg(F.sum('conversion')).collect()[0][0]
    print("The number of correct recommendation is: %d" % total_v)

    print(df.count())


if __name__ == '__main__':
    spark = SparkSession.builder.getOrCreate()
    # extract_top_N(5000)

    df = spark.read.json(INP_PATH)
    fp_growth(df)