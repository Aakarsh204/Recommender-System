import pandas as pd
from preprocess import Preprocessor

data = pd.read_csv('amazon.csv')
preprocessor = Preprocessor()
data = preprocessor.drop_columns(data)

data = preprocessor.cleaner(data)
data = preprocessor.add_column(data)
data = preprocessor.encode(data)

data.head()