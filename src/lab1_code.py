import json, re
from math import log, sqrt
from operator import add
from pyspark import SparkConf, SparkContext

INPUT_PATH = 'bookreviews.json'
OUTPUT_PATH = 'result.txt'

'''
Step 0: Pre-process the input data.

First count the total number of records, then generate a set for stopwords and a set for query words.
'''
total_len = 0
with open(INPUT_PATH) as f:
    for line in f:
        total_len += 1
print(total_len)
stopwords = set()
with open('stopwords.txt') as f:
    for line in f:
        stopwords.add(line.strip())

querywords = set()
with open('query.txt') as f:
    for line in f:
        querywords.update(line.strip().split())


'''
Step1：Compute frequency of every word in a document.
Output: (('word', 'doc'), count)

First build words_rdd to record every effective word in every doc. The key is a tuple consists of word and doc.
The step output is count_rdd, which counts the words in each doc.
'''
def readReviews(line):
    s = json.loads(line)
    doc_id = s['reviewerID'] + s['asin']
    doc_wd = []
    # Firstly, every word inside one text is converted to lower form;
    # Secondly, every separator r'[,.:;?!()]' is replaced by ' ', and then call split();
    # This brings huge convenience cause, if directly using re.split(), str like '.a' will be converted to ['', 'a']
    # Lastly remove stopwords.
    temp = (s['reviewText'] + s['summary']).lower()
    for word in re.sub(r'[,.:;?!()]', ' ', temp).split():
        if word not in stopwords:
            doc_wd.append((word, doc_id))
    return doc_wd

sc = SparkContext(master='local', conf=SparkConf())
words_rdd = sc.textFile(INPUT_PATH).flatMap(lambda x: readReviews(x))
count_rdd = words_rdd.map(lambda x: (x, 1)).reduceByKey(add)


'''
Step 2: Compute TF-IDF of every word w.r.t a document.
Output: (('word', 'doc'), tfidf)

First build a temp rdd for calculating df while reserving the previous counting info (see mergeValue)
The step output is tfidf_rdd, which calculates the tf-idf value for words in each doc.
'''
def mergeValue(value1, value2):
    # [('docA', 1)], [('docB', 2)] => [('docA', 1), ('docB', 2)]
    value1.extend(value2)
    return value1

def tf_idf(x):
    # x => (word, [('docA', 1), ('docB', 2)])
    output = []
    for t in x[1]:
        tfidf = (1.0 + log(t[1])) * log(total_len / len(x[1]))
        temp = ((x[0], t[0]), tfidf)
        output.append(temp)
    return output

a = count_rdd.map(lambda x: (x[0][0], [(x[0][1], x[1])]))
# Here we could also use the method below. Pls refer to YMT.
# a = a.combineByKey(s2_createCombiner, s2_mergeValue, s2_mergeCombiners)
tfidf_rdd = a.reduceByKey(mergeValue).flatMap(tf_idf)


'''
Step 3: Compute the normalized TF-IDF
Output: (('word', 'doc'), norm_tfidf)

Here the process is similar to Step 2. We first aggregate the TF-IDFs of words within the same docs.
The step output is norm_tfidf_rdd which stores the normalized TF-IDF values.
'''
def norm_tf_idf(x):
    # x => (doc, [('wordA', 1), ('wordB', 2)])
    output = []
    sum = 0
    for t in x[1]:
        sum += sqrt(t[1])
    for t in x[1]:
        temp = ((t[0], x[0]), t[1]/sum)
        output.append(temp)
    return output

a = tfidf_rdd.map(lambda x: (x[0][1], [(x[0][0], x[1])]))
norm_tfidf_rdd = a.reduceByKey(mergeValue).flatMap(norm_tf_idf)


"""
Step 4: Compute the relevance of every document w.r.t a query
Output: ('doc', relevance)

As the TF-IDFs are already normalized, we proved that the cosine relevance is equivalent to the process here.
filter() filters the RDD elements by query words first, and the elements are directly mapped to ('doc', norm_tfidf) and then reduced.
"""
a = norm_tfidf_rdd.filter(lambda x: x[0][0] in querywords)
query_res_rdd = a.map(lambda x: (x[0][1], x[1])).reduceByKey(add)


"""
Stage5: Sort and get top-k documents, here k=20.
Output: result.txt which contains the top 20 ('doc', relevance)

sortBy() is used to sort by relevance in descending order.
"""
final_res_rdd = query_res_rdd.sortBy(lambda x: x[1], ascending=False).take(20)

with open(OUTPUT_PATH, 'w') as f:
    for x in final_res_rdd:
        f.write(str(x) +'\n')
print(final_res_rdd)
sc.stop()
