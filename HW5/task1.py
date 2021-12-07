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
import binascii
from blackbox import BlackBox


def myhashs(s):
    result = []
    m = 69997
    p = 53591
    for j in range(hash_n):
        s_int = int(binascii.hexlify(s.encode('utf8')), 16)
        result.append(((a[j] * s_int + b[j]) % p) % m)
    return result


def check_in_set(x, hash_list):
    for h in hash_list:
        if filter_array[h] == 0:
            return (x, 1)
    return (x, 0)


def update_bloom_filter(hash_list):
    for h in hash_list:
        filter_array[h] = 1


def count_negative(x, l):
    if x in l:
        return 0
    else:
        return 1


hash_n = 2
a = []
b = []
for _ in range(hash_n):
    a.append(random.randint(10000, 100000))
    b.append(random.randint(10000, 100000))

if __name__ == '__main__':
    input_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_path = sys.argv[4]

    filter_array = [0] * 69997
    output_dict = {}
    bx = BlackBox()
    previous_set = set()

    start = time.time()
    sc = pyspark.SparkContext('local[*]', 'task1')
    sc.setLogLevel("WARN")

    current_batch = bx.ask(input_path, stream_size)
    current_hash = [(i, myhashs(i)) for i in current_batch]

    for cur in current_hash:
        update_bloom_filter(cur[1])
    previous_set = previous_set.union(set(current_batch))

    for i in range(num_of_asks):
        current_batch = bx.ask(input_path, stream_size)
        current_hash = [(i, myhashs(i)) for i in current_batch]

        false_positive_candidates = [check_in_set(i[0], i[1]) for i in current_hash]
        false_positive_candidates = [i[0] for i in false_positive_candidates if i[1] == 0]

        ground_t_negative = 0.0
        false_positive = 0.0

        for c in current_batch:
            if c not in previous_set:
                ground_t_negative += 1

        for c in false_positive_candidates:
            if c not in previous_set:
                false_positive += 1

        if ground_t_negative == 0 or false_positive == 0:
            output_dict[str(i)] = 0
        else:
            output_dict[str(i)] = false_positive / ground_t_negative

        for cur in current_hash:
            update_bloom_filter(cur[1])

        previous_set = previous_set.union(set(current_batch))

    q = open(output_path, "w")
    q.write("Time,FPR\n")
    for k, v in output_dict.items():
        q.write(str(k) + "," + str(v) + "\n")
    q.close()
    print("Duration:", time.time() - start)
