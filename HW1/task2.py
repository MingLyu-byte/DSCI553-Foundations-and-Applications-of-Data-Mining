import os
import sys

import pyspark
import json
import datetime
import time


def partition_by(ids, num=2, type_p="default"):
    if type_p == "default":
        RDD = ids
    else:
        RDD = ids.partitionBy(num, lambda x: partition_func(x) % num)
    return RDD, RDD.mapPartitions(lambda x: [sum(1 for i in x)]).collect()


def partition_func(String):
    return sum([ord(i) for i in String])


def top_10(usr_ids):
    return usr_ids.reduceByKey(lambda a, b: a + b).takeOrdered(10, key=lambda x: (-x[1], x[0]))


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    n_partition = int(sys.argv[3])

    sc = pyspark.SparkContext.getOrCreate()
    output = {}
    input_lines = sc.textFile(input_path, n_partition).map(lambda row: json.loads(row))
    default_partition_RDD, default_partition_size_list = partition_by(input_lines.map(lambda x: (x['business_id'], 1)),
                                                                      n_partition, "default")
    customized_partition_RDD, customized_partition_size_list = partition_by(
        input_lines.map(lambda x: (x['business_id'], 1)), n_partition, "customize")

    start = time.time()
    top_10(default_partition_RDD)
    end = time.time()
    default_time = end - start
    print(default_time, default_partition_size_list)
    output["default"] = {"n_partition": default_partition_RDD.getNumPartitions(),
                         "n_items": default_partition_size_list, "exe_time": default_time}

    start = time.time()
    top_10(customized_partition_RDD)
    end = time.time()
    customize_time = end - start
    print(customize_time, customized_partition_size_list)
    output["customized"] = {"n_partition": n_partition, "n_items": customized_partition_size_list,
                            "exe_time": customize_time}
    with open(output_path, 'w') as fp:
        json.dump(output, fp)
