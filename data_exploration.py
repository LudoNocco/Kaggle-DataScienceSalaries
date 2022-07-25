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
        if feature in ["work_year", "experience_level", "employment_type"]:
            df[feature].value_counts().plot(kind="bar")
            plt.savefig(f"graphs/{feature}_barplot.png")
            plt.close()
        
        elif feature == "job_title":
            df[feature].value_counts().sort_values(ascending=True)[-5:].plot(kind="barh")
            plt.savefig(f"graphs/{feature}_barplot.png")
            plt.close()
        
            # Extracting unfrequent job titles
            top_titles = ["head", "lead", "principal", "director"]
            print(top_titles)
            df["top_title"] = df.apply(lambda x: x["job_title"].lower().split(" ")[0] in top_titles, axis=1)
            df_top_titles = df[df["top_title"]==True]
            print("Number top level positions: ", df_top_titles.shape)
            print("Top level position average salary: ", df_top_titles["salary"].mean())

            low_titles = df[feature].value_counts().sort_values(ascending=True)[-5:].values
            print(low_titles)
            df["low_title"] = df.apply(lambda x: x["job_title"].lower().split(" ")[0] in low_titles, axis=1)
            df_low_titles = df[df["low_title"]==True]
            print("Number low level positions: ", df_low_titles.shape)
            print("Low level position average salary: ", df_low_titles["salary"].mean())

            import ipdb
            ipdb.set_trace()

        elif feature == "salary":
            #import ipdb
            #ipdb.set_trace()
            pass
        elif feature == "salary_currency":
            #import ipdb
            #ipdb.set_trace()
            pass
        elif feature == "salary_in_usd":
            #import ipdb
            #ipdb.set_trace()
            pass
        elif feature == "employee_residence":
            #import ipdb
            #ipdb.set_trace()
            pass
        elif feature == "remote_ratio":
            #import ipdb
            #ipdb.set_trace()
            pass
        elif feature == "company_location":
            #import ipdb
            #ipdb.set_trace()
            pass
        elif feature == "company_size":
            #import ipdb
            #ipdb.set_trace()
            pass

        
    
    return

if __name__=="__main__":
    data_path = "data/dsjs.csv"
    save_data_graphs(data_path)

