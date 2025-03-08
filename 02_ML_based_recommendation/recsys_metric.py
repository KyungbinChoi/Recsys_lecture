# 추천 metric function
import numpy as np

def precision_at_k(y_true, y_pred, k):
    # 예측 된 아이템 중 상위 k개만
    y_pred = y_pred[:k]
    # top-k item 중 유관 상품의 비율
    return len(set(y_true) & set(y_pred)) / k

def recall_at_k(y_true, y_pred, k):
    # 예측 된 아이템 중 상위 k개만
    y_pred = y_pred[:k]
    # 실제 유관 상품 중 상위 k개에 포함된 비율
    return len(set(y_true) & set(y_pred)) / len(y_true)

def mrr_at_k(y_true, y_pred, k):
    # 예측 된 아이템 중 상위 k개만
    y_pred = y_pred[:k]
    for i, p in enumerate(y_pred):
        if p in y_true:
            return 1 / (i+1)
    return 0

def average_precision_at_k(y_true, y_pred, k):
    # 예측 된 아이템 중 상위 k개만
    y_pred = y_pred[:k]
    # average precision at k
    score = 0.0
    num_hits = 0
    for i ,p in enumerate(y_pred):
        if p in y_true:
            num_hits += 1
            score += num_hits / (i+1)
    return score / min(len(y_true), k)

def ndcg_at_k(y_true, y_pred, k):
    # 예측 된 아이템 중 상위 k개만
    y_pred = y_pred[:k]
    # DCG at k
    dcg = sum([int(p in y_true) / np.log2(i+2) for i, p in enumerate(y_pred)])
    # IDCG at k
    idcg = sum([1 / np.log2(i+2) for i in range(min(len(y_true), k))])
    # NDCG at k
    return dcg / idcg