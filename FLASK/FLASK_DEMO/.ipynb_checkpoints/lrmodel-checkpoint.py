import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

class LRModel:
    def train(self):
        df = pd.read_csv('File/iris.csv')
        x = df.drop(['Id', 'Species'], axis = 1)
        y = df['Species']
        LR = LogisticRegression(random_state = 0).fit(x,y)
        with open('File/lr_model.pickle', 'wb') as files:
            pickle.dump(LR, files)
            
    def test(self, SL, SW, PL, PW):
        with open('File/lr_model.pickle', 'rb') as f:
            LR = pickle.load(f)
        y_pred = LR.predict([[SL, SW, PL, PW]])
        return y_pred[0]