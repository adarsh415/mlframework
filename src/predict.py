import os
import pandas as pd
import numpy as np
from sklearn import metrics
import joblib

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df['id'].values
    predictions = None

    for FOLD in range(5):
        df = pd.read_csv(TEST_DATA)
        encoder = joblib.load(os.path.join("models", f'{MODEL}_{FOLD}_label_encoder.pkl'))
        for c in encoder:
            df.loc[:, c] = encoder[c].transform(df[c].values.tolist())

        # data is ready for train
        cols = joblib.load(os.path.join("models", f'{MODEL}_{FOLD}_columns.pkl'))
        clf = joblib.load(os.path.join("models", f'{MODEL}_{FOLD}.pkl'))
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]
        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    predictions /= 5
    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=['id', 'target'])
    return sub


if __name__ == '__main__':
    submission = predict()
    submission.to_csv(f'models/{MODEL}.csv', index=False)