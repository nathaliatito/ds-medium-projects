import pickle

class run_scaler(object):

    def __init__(self):
        self.scaler = pickle.load(open("scaler.pickle","rb"))

    def data_preparation(self, df):
        to_transform_data = df.copy()
        to_transform_cols = to_transform_data.columns

        transformed = self.scaler.transform(to_transform_data.values)
        df.loc[:, to_transform_cols] = transformed

        return df