import os
import pyspark
import json
import datetime
import random
import math
import itertools
import time
import pandas as pd
import numpy as np
import sys
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from collections import Counter
from collections import defaultdict


def pearson_sim(b1, b2, business_dict):
    b1_scores = business_dict[b1]
    b2_scores = business_dict[b2]

    if not b1_scores or not b2_scores:
        return 0

    b1_dict = {k: v for (k, v) in b1_scores}
    b2_dict = {k: v for (k, v) in b2_scores}
    b1_avg = sum(b1_dict.values()) / len(b1_dict)
    b2_avg = sum(b2_dict.values()) / len(b2_dict)
    corated_items = set(b1_dict.keys()).intersection(set(b2_dict.keys()))
    C = 0.0
    A = 0.0
    B = 0.0
    if len(corated_items) < 75:
        return 0

    for item in corated_items:
        a = b1_dict[item] - b1_avg
        b = b2_dict[item] - b2_avg
        C += a * b
        A += a ** 2
        B += b ** 2

    if B == 0 or A == 0:
        return 0

    return C / (math.sqrt(A) * math.sqrt(B))


def calculate_score(neighbor_tuple, user, user_dict, n=3):
    s = 3
    if not neighbor_tuple:
        return s
    neighbor_tuple = filter(lambda x: x[0] > 0, neighbor_tuple)

    if not neighbor_tuple:
        return s
    neighbor_tuple = sorted(neighbor_tuple, key=lambda x: x[0], reverse=True)

    candidates = neighbor_tuple[0:min(len(neighbor_tuple), n)]

    B = sum([abs(k) for (k, v) in candidates])
    if B == 0:
        user_rating = user_dict[user]
        if not user_rating:
            return s
        else:
            return sum([v for (k, v) in user_rating]) / len(user_rating)
    else:
        A = sum([v * k for (k, v) in candidates])
        return A / B


if __name__ == '__main__':
    start = time.time()
    folder_path = sys.argv[1]
    input_test_path = sys.argv[2]
    output_path = sys.argv[3]

    sc = pyspark.SparkContext('local[*]', 'task1')
    sc.setLogLevel("WARN")

    # XGB model-based
    train_file_path = os.path.join(folder_path, "yelp_train.csv")
    train_rdd = sc.textFile(train_file_path).map(lambda x: x.strip().split(",")).filter(lambda x: x[0] != 'user_id')
    test_rdd = sc.textFile(input_test_path).map(lambda x: x.strip().split(",")).filter(lambda x: x[0] != 'user_id')

    user_dict = sc.textFile(os.path.join(folder_path, 'user.json')) \
        .map(lambda r: json.loads(r)) \
        .map(lambda r: (r['user_id'], [r['review_count'], r['average_stars']])) \
        .collectAsMap()

    business_dict = sc.textFile(os.path.join(folder_path, 'business.json')) \
        .map(lambda row: json.loads(row)) \
        .map(lambda row: (row['business_id'], [row['stars'], row['review_count'], row['longitude'], row['latitude']])) \
        .collectAsMap()

    xtrain = np.array(train_rdd.map(lambda row: np.array(user_dict[row[0]] + business_dict[row[1]])).collect())
    ytrain = np.array(train_rdd.map(lambda row: float(row[2])).collect())
    xgbr = xgb.XGBRegressor(eval_metric=['rmse'], max_depth=10, alpha=2, eta=0.2)
    xgbr.fit(xtrain, ytrain)
    xtest = np.array(test_rdd.map(lambda row: np.array(user_dict[row[0]] + business_dict[row[1]])).collect())
    ytest_xgbr = xgbr.predict(xtest)
    ids = test_rdd.map(lambda row: (row[0], row[1])).collect()
    asign = lambda t: 3 if np.isnan(t) or t == float("inf") or t == float("-inf") or not t else t
    ytest_xgbr = list(map(asign, ytest_xgbr))

    # CF item-based
    train_rdd = train_rdd.map(lambda x: (str(x[1]), str(x[0]), float(x[2])))
    test_rdd = test_rdd.map(lambda x: (str(x[1]), str(x[0])))

    train_user_distinct = train_rdd.map(lambda x: x[1]).distinct().collect()
    test_user_distinct = test_rdd.map(lambda x: x[1]).distinct().collect()
    user_distinct = list(set(train_user_distinct + test_user_distinct))
    user_to_index = {}
    for i, user in enumerate(user_distinct):
        user_to_index[user] = i

    train_business_distinct = train_rdd.map(lambda x: x[0]).distinct().collect()
    test_business_distinct = test_rdd.map(lambda x: x[0]).distinct().collect()
    business_distinct = list(set(train_business_distinct + test_business_distinct))
    business_to_index = {}
    for i, business in enumerate(business_distinct):
        business_to_index[business] = i

    output_dict = {}
    for i in range(len(ytest_xgbr)):
        user_id, business_id = ids[i]
        output_dict[(user_to_index[user_id], business_to_index[business_id])] = ytest_xgbr[i]

    index_to_user = {v: k for k, v in user_to_index.items()}
    index_to_business = {v: k for k, v in business_to_index.items()}

    user_dict = train_rdd.map(lambda x: (user_to_index[x[1]], (business_to_index[x[0]], x[2]))) \
        .groupByKey() \
        .map(lambda x: (x[0], list(x[1]))) \
        .collectAsMap()

    for user_id_index in user_to_index.values():
        if user_id_index not in user_dict.keys():
            user_dict[user_id_index] = []

    business_dict = train_rdd.map(lambda x: (business_to_index[x[0]], (user_to_index[x[1]], x[2]))) \
        .groupByKey() \
        .map(lambda x: (x[0], list(x[1]))) \
        .collectAsMap()

    for business_id_index in business_to_index.values():
        if business_id_index not in business_dict.keys():
            business_dict[business_id_index] = []

    test_predict = test_rdd.map(lambda x: (user_to_index[x[1]], business_to_index[x[0]])) \
        .map(lambda x: (x[0], x[1], user_dict[x[0]])) \
        .map(lambda x: (
    x[0], x[1], [(pearson_sim(x[1], business_id, business_dict), score) for (business_id, score) in x[2]]))

    test_predict = test_predict.map(lambda x: (x[0], x[1], calculate_score(x[2], x[0], user_dict, 2))) \
        .map(lambda x: (x[0], x[1], x[2] * 0.1 + output_dict[(x[0], x[1])] * 0.9)) \
        .map(lambda x: (index_to_user[x[0]], index_to_business[x[1]], x[2])) \
        .collect()

    f = open(output_path, "w")
    f.write("user_id, business_id, prediction\n")
    for i in test_predict:
        f.write(",".join([str(j) for j in i]) + "\n")
    f.close()
    print("Duration:", time.time() - start)