import pandas as pd
from scipy.stats import spearmanr

def analyse_data(dataframe_path:str):
    df = pd.read_csv(data_path)
    salary = df["salary"]
    explevel = df["experience_level"]

    correlation = spearmanr(salary, explevel)
    print(correlation)

if __name__=="__main__":
    data_path = "data/dsjs.csv"
    analyse_data(data_path)