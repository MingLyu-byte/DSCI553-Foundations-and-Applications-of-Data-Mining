import os
import sys

import pyspark
import json
import datetime


def total_num_reviews(review_ids):
    return review_ids.count()


def total_num_reviews_2018(review_ids_date):
    return review_ids_date.filter(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S').year == 2018).count()


def total_num_distinct(ids):
    return ids.distinct().count()


def top_10(ids):
    return ids.reduceByKey(lambda a, b: a + b).takeOrdered(10, key=lambda x: (-x[1], x[0]))


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    sc = pyspark.SparkContext.getOrCreate()

    output_dict = {}
    input_lines = sc.textFile(input_path).map(lambda row: json.loads(row))
    output_dict['n_review'] = total_num_reviews(input_lines.map(lambda x: x['review_id']))
    output_dict['n_review_2018'] = total_num_reviews_2018(input_lines.map(lambda x: x['date']))
    output_dict['n_user'] = total_num_distinct(input_lines.map(lambda x: x['user_id']))
    output_dict['top10_user'] = top_10(input_lines.map(lambda x: (x['user_id'], 1)))
    output_dict['n_business'] = total_num_distinct(input_lines.map(lambda x: x['business_id']))
    output_dict['top10_business'] = top_10(input_lines.map(lambda x: (x['business_id'], 1)))
    with open(output_path, 'w') as fp:
        json.dump(output_dict, fp)