import numpy as np
from sklearn.metrics import accuracy_score
import json
import os


class CustomMetrics:
    def __init__(self, label_ids):
        self.LABEL_IDS = label_ids

    def compute_metrics(self, pred):
        logits, labels = pred
        preds = logits.argmax(axis=-1)
        label_tokens_ids = np.array(self.LABEL_IDS)
        index_mapping = {value.item(): idx for idx, value in enumerate(label_tokens_ids)}
        labels = labels[np.isin(labels, label_tokens_ids)]
        labels = np.array([index_mapping[label.item()] for label in labels])
        acc = accuracy_score(labels, preds)
        return {'accuracy': acc}
    

def get_directory_path(file_path):
    # 파일의 디렉터리 경로를 추출
    directory_path = os.path.dirname(file_path)
    return directory_path


def merge_data(path1, path2):
    print("merge dataset is not exist. This program will create merge.json")
    # 첫 번째 JSON 파일 읽기
    with open(path1, 'r', encoding="utf-8") as file:
        json_data1 = json.load(file)

    # 두 번째 JSON 파일 읽기
    with open(path2, 'r', encoding="utf-8") as file:
        json_data2 = json.load(file)

    # 두 JSON 데이터를 병합
    merged_data = []
    for i in json_data1:
        merged_data.append(i)

    for i in json_data2:
        merged_data.append(i)

    directory_path = get_directory_path(path1)
    with open(os.path.join(directory_path, 'merge.json'), "w", encoding="utf-8") as f:
        f.write(json.dumps(merged_data, ensure_ascii=False, indent=4))
        
    print("두 JSON 파일이 성공적으로 병합되었습니다.")