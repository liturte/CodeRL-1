import pickle

# 指定你的文件路径
file_path = '/data/coding/CodeRL/outputs/python_result/15.pkl'

# 读取PKL文件内容
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# 打印读取到的数据
print(data)
