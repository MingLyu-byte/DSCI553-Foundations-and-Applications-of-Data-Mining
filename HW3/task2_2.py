import pandas as pd
import numpy as np
import os
import sys
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pyspark
import json
import time

if __name__ == '__main__':
    folder_path = sys.argv[1]
    input_test_path = sys.argv[2]
    output_path = sys.argv[3]

    start = time.time()
    conf = pyspark.SparkConf()
    sc = pyspark.SparkContext.getOrCreate(conf)
    sc.setLogLevel("WARN")
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

    ypred = xgbr.predict(xtrain)
    mse = mean_squared_error(ytrain, ypred)
    print("MSE: %.2f" % mse)
    print("RMSE: %.2f" % (mse ** (1 / 2.0)))

    xtest = np.array(test_rdd.map(lambda row: np.array(user_dict[row[0]] + business_dict[row[1]])).collect())
    ytest = xgbr.predict(xtest)
    ids = test_rdd.map(lambda row: (row[0], row[1])).collect()
    asign = lambda t: 3 if np.isnan(t) or t == float("inf") or t == float("-inf") or not t else t
    ytest = list(map(asign, ytest))

    f = open(output_path, "w")
    f.write("user_id, business_id, prediction\n")
    for i in range(len(ytest)):
        f.write(",".join(ids[i]) + "," + str(ytest[i]) + "\n")
    print("Duration:", time.time() - start)
