from sklearn.metrics import f1_score

def accuracy_function(real, pred):    
    score = f1_score(real, pred, average='macro')
    return score