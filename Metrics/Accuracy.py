from sklearn.metrics import f1_score


class F1:
    def __init__(self, dataset=None):
        self.dataset = dataset
        
    def __call__(self, *args: Any, **kwds: Any):
        f1_score(average=None)
        
        
        