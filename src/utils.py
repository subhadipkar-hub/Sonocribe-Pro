from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    try:
        auc = roc_auc_score(
            torch.nn.functional.one_hot(torch.tensor(y_true)),
            torch.nn.functional.one_hot(torch.tensor(y_pred))
        )
    except:
        auc = float('nan')
    return acc, f1, auc
