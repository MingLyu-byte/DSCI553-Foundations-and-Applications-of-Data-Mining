import os
import sys

import pyspark
import json
import datetime
import random
import math
import itertools
import time
from collections import Counter
from collections import defaultdict
import graphframes
from pyspark.sql import SQLContext

if __name__ == '__main__':
    input_thresh_hold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    output_path = sys.argv[3]

    start = time.time()
    sc = pyspark.SparkContext.getOrCreate()
    sc.setLogLevel("WARN")
    start = time.time()
    input_lines = sc.textFile(input_file_path).map(lambda x: x.strip().split(",")) \
        .filter(lambda x: x[0] != 'user_id') \
        .map(lambda x: (str(x[0]), str(x[1])))

    distinct_user = input_lines.map(lambda x: x[0]).distinct().collect()
    distinct_user_t = [(i,) for i in distinct_user]
    sqlContext = SQLContext(sc)
    vertices = sqlContext.createDataFrame(distinct_user_t).toDF("id")

    user_dict = input_lines.groupByKey() \
        .collectAsMap()

    candidates = set()
    for c in itertools.combinations(distinct_user, 2):
        p1 = c[0]
        p2 = c[1]
        if len(set(user_dict[p1]).intersection(set(user_dict[p2]))) >= input_thresh_hold:
            candidates.add((p1, p2))

    candidates = list(candidates)
    edges = sqlContext.createDataFrame(candidates).toDF("src", "dst")
    g = graphframes.GraphFrame(vertices, edges)
    result = g.labelPropagation(maxIter=5)

    groups_rdd = result.rdd.coalesce(1) \
        .map(lambda row: (row[1], row[0])) \
        .groupByKey() \
        .map(lambda row: sorted(list(row[1]))) \
        .sortBy(lambda row: (len(row), row)) \
        .collect()

    f = open(output_path, "w")
    for i in groups_rdd:
        f.write(",".join(i) + "\n")

    print("Duration:", time.time() - start)