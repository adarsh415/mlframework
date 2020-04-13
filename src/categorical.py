from sklearn import preprocessing

"""
- label encoding
- binary encoding
- one hot encoding
"""

class CategoricalFeatures:
    def __init__(self, df, cat_features, enc_type, handle_na=False ):
        """
        df - pandas dataframe
        categorical_features - list of column names
        encoding_type - type of encoding
        """
        self.df = df
        self.cat_features = cat_features
        self.enc_type = enc_type
        self.label_encoder = dict()
        self.binary_encoder = dict()
        self.ohe = None
        self.handle_na = handle_na
        
        if self.handle_na:
            for cat in self.cat_features:
                self.df.loc[:,cat] = self.df.loc[:,cat].astype('str').fillna('-9999999')
        self.output_df = self.df.copy(deep=True)
        
    def _label_encoding(self):
        for c in self.cat_features:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoder[c] = lbl
        return self.output_df

    def _label_binarization(self):
        for c in self.cat_features:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values)
            self.output_df = self.output_df.drop(c, axis=1)
            for j in range(val.shape[1]):
                new_col = c+'_'+f'__bin__{j}'
                self.output_df[new_col] = val[:, j]
            self.binary_encoder[c] = lbl
        return self.output_df

    def _label_onehot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_features].values)
        self.ohe = ohe
        return ohe.transform(self.df[self.cat_features].values)


    def fit_transform(self):
        if self.enc_type == 'label':
            return self._label_encoding()
        elif self.enc_type == 'binary':
            return self._label_binarization()
        elif self.enc_type == 'onehot':
            return self._label_onehot()
        else:
            raise Exception("unknown encoding")

    def transform(self, dataframe):
        if self.handle_na:
            for cat in self.cat_features:
                dataframe.loc[:,cat] = dataframe.loc[:,cat].astype('str').fillna('-9999999')
        if self.enc_type == 'label':
            for c, lbl in self.label_encoder:
                dataframe.loc[:c] = lbl.transform(dataframe[c].values)
            return dataframe
        elif self.enc_type == 'binary':
            for c, lbl in self.binary_encoder:
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)
                for j in range(val.shape[1]):
                    new_col = c+'_'+f'__bin__{j}'
                    dataframe[new_col] = val[:, j]
            return dataframe
        elif self.enc_type == 'onehot':
            return self.ohe.transform(dataframe[self.cat_features].values)
        else:
            raise Exception("unknown encoding")

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('${workspaceRoot}/../input/train_cat.csv').head(100)
    df_test = pd.read_csv('${workspaceRoot}/../input/test_cat.csv').head(100)

    train_idx = df['id'].values
    test_idx = df_test['id'].values

    df_test['target'] = -1
    full_df = pd.concat([df, df_test])
    cols = [c for c in full_df.columns if c not in ['id', 'target']]
    cat_feat = CategoricalFeatures(
        full_df,
        cols,
        'onehot'
    )
    print(full_df.shape)
    full_data_transformed = cat_feat.fit_transform()
    print(full_data_transformed.shape)
    #print(cat_feat.fit_transform().shape())
    # train_df = full_data_transformed[full_data_transformed['id'].isin(train_idx)].reset_index(drop=True)
    # test_df = full_data_transformed[full_data_transformed['id'].isin(test_idx)].reset_index(drop=True)
    # print(train_df.head())
    # print(test_df.head())

