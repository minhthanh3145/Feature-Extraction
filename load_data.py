from pandas import read_csv


def load_data():
    # load data
    df = read_csv('./dataset/pima-indians-diabetes.data.csv',
                  names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class'])
    array = df.values
    X = array[:, 0:8]
    Y = array[:, 8]
    return X, Y
