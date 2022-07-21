import os
import pandas as pd
import matplotlib.pyplot as plt

def save_data_graphs(dataframe_path: str):
    df = pd.read_csv(dataframe_path)
    #print(df.columns)
    #df["salary"].hist()
    #plt.show()
    
    #Create folder to store graphs
    os.makedirs("graphs", exist_ok=True)
    
    features = df.columns
    for feature in features:
        if feature in ["work_year", "experience_level"]:
            df[feature].value_counts().plot(kind="bar")
            plt.savefig(f"graphs/{feature}_barplot.png")

        elif feature == "experience_level":
            import ipdb
            ipdb.set_trace()
        
        elif feature == "employment_type":
            pass
        elif feature == "job_title":
            pass
        elif feature == "salary":
            pass
        elif feature == "salary_currency":
            pass
        elif feature == "salary_in_usd":
            pass
        elif feature == "employee_residence":
            pass
        elif feature == "remote_ratio":
            pass
        elif feature == "company_location":
            pass
        elif feature == "company_size":
            pass
    
    return

if __name__=="__main__":
    data_path = "data/dsjs.csv"
    save_data_graphs(data_path)

