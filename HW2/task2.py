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


def generate_basket(ids, filter_thresh):
    result = ids.groupByKey().mapValues(list) \
        .filter(lambda row: len(row[1]) > filter_thresh) \
        .map(lambda row: sorted(list(set(row[1]))))
    return result


def generate_candidates(array, candidates_prev, size):
    output = {}
    for ar in array:
        if size == 1:
            for item in ar:
                item = (item,)
                output[item] = output.get(item, 0) + 1
        else:
            intersection = set(ar).intersection(candidates_prev)
            intersection = sorted(list(intersection))
            for item in itertools.combinations(intersection, size):
                output[item] = output.get(item, 0) + 1
    return output


def candidates_filter(count_dict, ps):
    output = []
    for item in count_dict:
        if count_dict[item] >= ps:
            output.append(item)
    return output


def map_partition_find_candidate_func(partition, support, size):
    output = []
    partition = list(partition)
    p = len(partition) / size
    ps = math.ceil(p * support)

    i = 1
    candidates = generate_candidates(partition, [], i)
    final_candidates = candidates_filter(candidates, ps)
    if final_candidates:
        output.append(final_candidates)
    if len(final_candidates) <= 1:
        return output
    final_candidates = set([i[0] for i in final_candidates])

    i += 1
    while True:
        candidates = generate_candidates(partition, final_candidates, i)
        final_candidates = candidates_filter(candidates, ps)

        if final_candidates:
            output.append(final_candidates)
        if len(final_candidates) <= 1:
            break

        final_candidates = set([i for ar in final_candidates for i in ar])
        i += 1

    return output


def map_partition_find_frequent_func(partition, candidates):
    output = {}
    for part in partition:
        for candidate in candidates:
            if set(candidate).issubset(set(part)):
                output[candidate] = output.get(candidate, 0) + 1
    return [(k, v) for k, v in output.items()]


def output_format(array,f):
    array = [list(g) for k, g in itertools.groupby(array, key=len)]
    f.write(",".join([str(i).replace(",","") for i in array[0]]) + "\n\n")
    for i in range(1,len(array)):
        f.write(",".join([str(j) for j in array[i]]) + "\n\n")


def preprocess(input_file, output_file, filter_thresh):
    f = open(input_file, "r", encoding='utf-8')
    w = open(output_file, "w")
    lines = f.readlines()
    f.close()
    lines = [line.strip().split(",") for line in lines[1:]]
    lines = [[i[0][1:-1], i[1][1:-1].lstrip("0"), i[5][1:-1].lstrip("0")] for i in lines]
    lines = [[i[0] + "-" + i[1], i[2]] for i in lines]
    w.write("DATE-CUSTOMER_ID,PRODUCT_ID\n")
    for line in lines:
        w.write(",".join(line) + "\n")
    w.close()
    return


if __name__ == '__main__':
    filter_thresh = int(sys.argv[1])
    support = int(sys.argv[2])
    input_path = sys.argv[3]
    output_path = sys.argv[4]
    intermediate = "customer_product.csv"
    preprocess(input_path, intermediate, filter_thresh)

    sc = pyspark.SparkContext.getOrCreate()
    sc.setLogLevel("WARN")
    start = time.time()
    input_lines = sc.textFile(intermediate).map(lambda row: row.strip().split(","))
    header = input_lines.first()
    input_lines = input_lines.filter(lambda line: line != header)
    q = open(output_path, "w")
    q.write("Candidates:\n")

    basket_rdd = generate_basket(input_lines.map(lambda row: (str(row[0]), str(row[1]))), filter_thresh)
    num_basket = basket_rdd.count()

    # phase 1 mapreduce
    candidates_p1 = basket_rdd.mapPartitions(
        lambda partition: map_partition_find_candidate_func(partition, support, num_basket)) \
        .flatMap(lambda x: x) \
        .distinct() \
        .sortBy(lambda x: (len(x), x)) \
        .collect()
    output_format(candidates_p1, q)

    # phase 2 mapreduce
    frequent_p2 = basket_rdd.mapPartitions(lambda partition: map_partition_find_frequent_func(partition, candidates_p1)) \
        .reduceByKey(lambda a, b: a + b) \
        .filter(lambda x: x[1] >= support) \
        .map(lambda x: x[0]) \
        .sortBy(lambda x: (len(x), x)) \
        .collect()

    q.write("Frequent Itemsets:\n")
    output_format(frequent_p2, q)

    print("Duration:", time.time() - start)
