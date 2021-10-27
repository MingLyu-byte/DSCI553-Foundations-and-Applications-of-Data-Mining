import os
import sys

import pyspark
import json
import datetime
import time


def top_10(rdd, n):
    agg_rdd = rdd.aggregateByKey((0, 0), lambda key, val: (key[0] + val, key[1] + 1),
                                 lambda t1, t2: (t1[0] + t2[0], t1[1] + t2[1]))
    if n == 0:
        return agg_rdd.map(lambda x: (x[0], (x[1][0] / x[1][1]))).sortBy(lambda x: (-x[1], x[0])).collect()
    else:
        return agg_rdd.map(lambda x: (x[0], (x[1][0] / x[1][1]))).takeOrdered(n, lambda x: (-x[1], x[0]))


if __name__ == '__main__':
    review_filepath = sys.argv[1]
    business_filepath = sys.argv[2]
    output_path_a = sys.argv[3]
    output_path_b = sys.argv[4]

    sc = pyspark.SparkContext.getOrCreate()

    output = {}
    start = time.time()
    input_lines_review = sc.textFile(review_filepath).map(lambda row: json.loads(row))
    input_lines_business = sc.textFile(business_filepath).map(lambda row: json.loads(row))
    city_review = input_lines_review.map(lambda x: (x['business_id'], x['stars'])). \
        join(input_lines_business.map(lambda x: (x['business_id'], x['city']))).map(lambda x: (x[1][1], x[1][0]))
    result = top_10(city_review, 10)
    print(result)
    end = time.time()
    output['m2'] = end - start

    result = top_10(city_review, 0)
    f = open(output_path_a, "w")
    f.write("city,stars" + "\n")
    for i in range(len(result)):
        data = result[i]
        f.write(data[0] + "," + str(data[1]) + "\n")

    start = time.time()
    input_lines_review = sc.textFile(review_filepath).map(lambda row: json.loads(row))
    input_lines_business = sc.textFile(business_filepath).map(lambda row: json.loads(row))
    city_review = input_lines_review.map(lambda x: (x['business_id'], x['stars'])). \
        join(input_lines_business.map(lambda x: (x['business_id'], x['city']))).map(lambda x: (x[1][1], x[1][0]))
    agg_rdd = city_review.aggregateByKey((0, 0), lambda key, val: (key[0] + val, key[1] + 1),
                                         lambda t1, t2: (t1[0] + t2[0], t1[1] + t2[1]))
    city_rate_list = agg_rdd.map(lambda x: (x[0], (x[1][0] / x[1][1]))).collect()
    city_rate_list = sorted(city_rate_list, key=lambda x: (-x[1], x[0]))
    city_rate_list = city_rate_list[0:10]
    print(city_rate_list)
    end = time.time()
    output['m1'] = end - start
    output['reason'] = "Python sort is faster since Spark sorting requires shuffling, which is computational expensive."
    with open(output_path_b, 'w') as fp:
        json.dump(output, fp)
    print(output)
