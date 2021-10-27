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

if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    start = time.time()
    hash_n = 100
    band = 100
    r = 1
    sc = pyspark.SparkContext.getOrCreate()
    sc.setLogLevel("WARN")
    start = time.time()
    input_lines = sc.textFile(input_path).map(lambda row: row.strip().split(","))
    header = input_lines.first()
    input_lines = input_lines.filter(lambda line: line != header)
    review_rdd = input_lines.map(lambda row: (str(row[0]), str(row[1])))

    distinct_users = input_lines.map(lambda row: row[0]).distinct().collect()
    users_dict = {}
    for index, user in enumerate(distinct_users):
        users_dict[user] = index

    users_index = review_rdd.map(lambda x: (x[1], users_dict[x[0]]))
    hashed_user_index = defaultdict(list)

    for j in range(hash_n):
        a = random.randint(100, 100000)
        b = random.randint(100, 100000)
        p = 53591
        m = 44963
        for i in range(len(distinct_users)):
            hashed_user_index[i].append(((a * i + b) % p) % m)

    signature_m = users_index.groupByKey() \
        .map(lambda row: (row[0], list(set(row[1])))) \
        .map(lambda row: (row[0], [hashed_user_index[user] for user in row[1]])) \
        .map(lambda row: (row[0], [min(c) for c in zip(*row[1])])) \
        .collect()

    candidates = set()
    for b in range(band):
        cur_bucket = defaultdict(set)
        for s in signature_m:
            cur_start_index = b * r
            output = ""
            for i in range(r):
                output += str(s[1][cur_start_index + i])
            bucket_hash = hash(output)
            cur_bucket[bucket_hash].add(s[0])
        for k, v in cur_bucket.items():
            if len(v) >= 2:
                for p in itertools.combinations(v, 2):
                    candidates.add(tuple(sorted(list(p))))

    f = open(output_path, "w")
    f.write("business_id_1, business_id_2, similarity\n")

    b_u_dict = users_index.groupByKey() \
        .map(lambda row: (row[0], set(row[1]))) \
        .collectAsMap()

    output = []
    for candidate in candidates:
        x1 = candidate[0]
        x2 = candidate[1]
        x1_users = b_u_dict[x1]
        x2_users = b_u_dict[x2]
        similarity = float(len(x1_users.intersection(x2_users)) / len(x1_users.union(x2_users)))
        if similarity >= 0.5:
            output.append((x1, x2, str(similarity)))
    output.sort()
    for row in output:
        f.write(",".join(row) + "\n")
    f.close()
    print("Duration:", time.time() - start)