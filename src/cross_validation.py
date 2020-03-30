from sklearn import model_selection
import pandas as pd

class CrossValidation:
    def __init__(
            self,
            df,
            target_cols,
            problem_type='binary_classification',
            n_folds=5,
            shuffle=True,
            random_state=42
    ):
        self.df = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type= problem_type
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state

        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.df['kfold'] = -1

    def split(self):
        if self.problem_type == 'binary_classification':
            unique_values = self.df[self.target_cols[0]].nunique()
            if unique_values == 1:
                raise Exception("only 1 unique value found")
            elif unique_values > 1:
                target = self.target_cols[0]
                kf = model_selection.StratifiedKFold(n_splits=self.n_folds,
                                                     shuffle=False,
                                                     random_state=self.random_state
                                                     )
                for fold, (train_idx, valid_idx) in enumerate(kf.split(X=self.df,
                                                                       y=self.df[target].values)):
                    self.df.loc[valid_idx, 'kfold'] = fold
        return self.df


if __name__ == '__main__':
    df = pd.read_csv('../input/train.csv')
    cv = CrossValidation(df, target_cols=["target"])
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())
