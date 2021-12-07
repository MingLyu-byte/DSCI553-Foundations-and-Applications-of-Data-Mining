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
    m = 1000
    p = 53591
    for j in range(hash_n):
        s_int = int(binascii.hexlify(s.encode('utf8')),16)
        hash_v = ((a[j] * s_int + b[j]) % p) % m
        binary = bin(hash_v)
        trailing_zero = len(binary) - len(binary.rstrip("0"))
        result.append(trailing_zero)
    return result


hash_n = 200
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

    start = time.time()
    sc = pyspark.SparkContext('local[*]', 'task2')
    sc.setLogLevel("WARN")

    output_dict = {}
    bx = BlackBox()

    for i in range(num_of_asks):
        current_batch = bx.ask(input_path, stream_size)
        current_hash = [myhashs(i) for i in current_batch]

        cur_trailing_zero = []
        for k in range(hash_n):
            cur_trailing_zero.append(max([j[k] for j in current_hash]))

        output_dict[i] = ((len(set(current_batch))), int(sum([2 ** p for p in cur_trailing_zero]) / hash_n))

    q = open(output_path, "w")
    q.write("Time,Ground Truth,Estimation\n")
    for k, v in output_dict.items():
        q.write(str(k) + "," + str(v[0]) + "," + str(v[1]) + "\n")
    q.close()
    print("Duration:", time.time() - start)