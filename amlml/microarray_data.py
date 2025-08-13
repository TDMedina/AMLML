
import pandas as pd

def read_microarray_table(input_file):
    data = pd.read_csv(input_file, sep="\t", index_col=0)
    data = data.T
    data.index.names = ["sample"]
    return data
