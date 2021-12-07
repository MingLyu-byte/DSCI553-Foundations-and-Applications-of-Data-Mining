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
import copy

if __name__ == '__main__':
    random.seed(553)
    input_path = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_path = sys.argv[4]

    start = time.time()
    sc = pyspark.SparkContext('local[*]', 'task3')
    sc.setLogLevel("WARN")

    output_dict = {}
    bx = BlackBox()
    n = 100

    current_batch = bx.ask(input_path, stream_size)
    user_batch = copy.deepcopy(current_batch)

    output_dict[100] = (user_batch[0], user_batch[20], user_batch[40], user_batch[60], user_batch[80])

    for i in range(2, num_of_asks + 1):
        current_batch = bx.ask(input_path, stream_size)

        for user in current_batch:
            n += 1
            p1 = 100 / n
            p = random.random()
            if p < p1:
                index = random.randint(0, 99)
                user_batch[index] = user

        output_dict[i * 100] = (user_batch[0], user_batch[20], user_batch[40], user_batch[60], user_batch[80])

    q = open(output_path, "w")
    q.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")
    for k, v in output_dict.items():
        q.write(str(k) + "," + ",".join(v) + "\n")
    q.close()
    print("Duration:", time.time() - start)