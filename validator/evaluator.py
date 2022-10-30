from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate(golden_lists, predict_lists):
    acc = accuracy_score(golden_lists, predict_lists)
    pre = precision_score(golden_lists, predict_lists, average='macro')
    rec = recall_score(golden_lists, predict_lists, average='macro')
    f1 = f1_score(golden_lists, predict_lists, average='macro')
    return acc, pre, rec, f1
