import os

cur_path = os.path.split(os.path.realpath(__file__))[0]
bert_data_path = os.path.join(cur_path, 'data')
bert_model_path = os.path.join(cur_path, 'models')

if __name__ == "__main__":
    print(cur_path)
