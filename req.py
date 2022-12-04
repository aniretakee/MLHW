import requests
import pandas as pd

# url = 'http://127.0.0.1:8000/predict_item'
#
# test_path = 'cars_test.csv'
# df_test = pd.read_csv(test_path).sample(frac=1).head(1)
# param_add = df_test.to_dict()
#
# print(param_add)
# post_ = requests.post(url, json=param_add)
# print(post_.text)




url = 'http://127.0.0.1:8000/predict_items'
test_path = 'cars_test.csv'
df = pd.read_csv(test_path)
df = df[~df.isna().any(axis=1)].sample(frac=1).head(10)

param_add = df.to_dict()

print(param_add)
post_ = requests.post(url, json=param_add)
print(post_.text)


