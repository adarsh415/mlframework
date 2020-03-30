from sklearn import model_selection
import pandas as pd

"""
- single column regression
- multi column regression
- holdout
- binary classification
- multiclass classification
-multi label classification
"""

class CrossValidation:
    def __init__(
            self,
            df,
            target_cols,
            shuffle,
            problem_type='binary_classification',
            multilabel_delimiter=',',
            n_folds=5,
            random_state=42
    ):
        self.df = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type= problem_type
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.multilabel_delimiter = multilabel_delimiter

        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.df['kfold'] = -1

    def split(self):
        if self.problem_type in ('binary_classification', 'multiclass_classification'):
            if self.num_targets != 1:
                raise Exception("Invalid number for this problem type")
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
        elif self.problem_type in ('single_col_regression', 'multi_col_regression'):
            if self.num_targets != 1 and self.problem_type == 'single_col_regression':
                raise Exception("Invalid number for this problem type")
            if self.num_targets < 2 and self.problem_type == 'multi_col_regression':
                raise Exception("Invalid number for this problem type")
            kf = model_selection.KFold(n_splits=self.n_folds)
            for fold, (train_idx, valid_idx) in enumerate(kf.split(X=self.df)):
                self.df.loc[valid_idx, 'kfold'] = fold
        elif self.problem_type.startswith('holdout_'):
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.df)*holdout_percentage/100)
            print(num_holdout_samples)
            self.df.loc[:num_holdout_samples,'kfold']=0
            self.df.loc[num_holdout_samples:, 'kfold']=1

        elif self.problem_type == 'multilabel_classification':
            if self.num_targets != 1:
                raise Exception('Invalid number for target for this problem type')
            targets = self.df[self.target_cols[0]].apply(lambda x: len(x.str.split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.n_folds,
                                                 shuffle=False,
                                                 random_state=self.random_state
                                                 )
            for fold, (train_idx, valid_idx) in enumerate(kf.split(X=self.df,
                                                                   y=targets)):
                self.df.loc[valid_idx, 'kfold'] = fold
        else:
            raise Exception("Problem type not understood")

        return self.df


if __name__ == '__main__':
    df = pd.read_csv('../input/train.csv')
    cv = CrossValidation(df, target_cols=["target"],shuffle=False, problem_type='holdout_10')
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())
