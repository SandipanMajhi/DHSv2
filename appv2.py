import streamlit as st
import json
# from streamlit_extras.app_logo import add_logo

import pandas as pd
from constants import *
from  st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode, ColumnsAutoSizeMode

import pickle as pkl

st.set_page_config(layout="wide")

st.sidebar.title("Parameters")

add_selectbox = st.sidebar.selectbox(
    label= "Ranking Type",
    options= ["Normal-v1", "Reasoning-v1.5", "cosine-similarity-v1.5"]
)


col1, mid, col2 = st.columns([1, 0.5, 30])
with col1:
    st.image("Assets/sokat.jpeg", width=60)
with col2:
    st.header('DHS AI Use Case Ranking', divider='rainbow') 

st.write('Powered by LLAMA-2-70B, Mixtral-8X7B, GPT-3.5 and GPT-4.')

@st.cache_data
def load_data():
    df = pd.read_csv("./Data/Streamlit_pass1.csv")
    df.rename(columns= {"gpt-4-turbo-preview_summaries" : "Summaries"}, inplace=True)

    #### Reasoning Based Ranks ####

    with open("outputv1.json", "r") as fp:
        data_risks = json.load(fp)

    df1 = pd.DataFrame.from_dict(data_risks['explainable'])
    df1["explainable_score"] = df1["pos_score"] - df1["neg_score"]
    df1 = df1[["Use Case ID", "Use Case Name", "explainable_score"]]

    df2 = pd.DataFrame.from_dict(data_risks['trustworthy'])    
    df2["trustworthy_score"] = df2["pos_score"] - df2["neg_score"]
    df2 = df2[["Use Case ID", "Use Case Name", "trustworthy_score"]]

    df1 = df1.merge(df2, on = ["Use Case ID", "Use Case Name"])

    for i in range(len(risk_classes)):
        df1[f"{risk_classes[i]}_score"] = df1[f"{risk_classes[i]}_score"].rank(ascending=False).astype('Int64')


    #### Cosine Sim Based Ranks #####

    with open("outputsv1.5_cosinescore.json", "r") as fp:
        cdata = json.load(fp)


    df_cosine1 = pd.DataFrame.from_dict(cdata['explainable'])
    df_cosine1 = df_cosine1[["Use Case ID", "Use Case Name", "cosine_explainable_score"]]

    df_cosine2 = pd.DataFrame.from_dict(cdata['trustworthy'])
    df_cosine2 = df_cosine2[["Use Case ID", "Use Case Name", "cosine_trustworthy_score"]]

    df_cosine1 = df_cosine1.merge(df_cosine2, on = ["Use Case ID", "Use Case Name"])
    
    for i in range(len(risk_classes)):
        df_cosine1[f"cosine_{risk_classes[i]}_score"] = df_cosine1[f"cosine_{risk_classes[i]}_score"].rank(ascending=False).astype('Int64')

    return df, df1, df_cosine1

@st.cache_data
def load_corr_data():
    df = pd.read_csv("./Data/")

def styler_frame():
    d = {
        "Use Case ID" : "background-color:#E7E8D1",
        "Use Case Name" : "background-color:#E7E8D1"
        # "BIAS_Ranking" : "background-color:#E7E8D1",
        # "COMPLEXITY_Ranking" : "background-color:#FFF2D7",
        # "EFFECTIVENESS_Ranking" : "background-color:#F96167",
        # "EXPENSIVENESS_Ranking" : "background-color:#F9E795",
        # "EXPLAINABLE_Ranking" : "background-color:#D3C5E5",
        # "SAFE_Ranking" : "background-color:#F7C5CC",
        # "TRUSTWORTHY_Ranking" : "background-color:#A7BEAE"
    }

    return d

def background_cell(x):
    d = styler_frame()
    s = pd.DataFrame(d,index=x.index,columns=x.columns)
    return s


#### Main Code #####

main_df, dfv2, dfcosv2 = load_data()

if add_selectbox == "Normal-v1":
    st.dataframe(
        # main_df.style.apply(background_cell,axis=None),
        main_df.style.background_gradient(subset=["BIAS_Ranking", "COMPLEXITY_Ranking",
                                                "EFFECTIVENESS_Ranking", "EXPENSIVENESS_Ranking", "EXPLAINABLE_Ranking",
                                                "SAFE_Ranking", "TRUSTWORTHY_Ranking" ],cmap='YlOrRd'),
        use_container_width= True,
        height = 700,
        hide_index=True,
        column_order= ("Use Case ID", "Use Case Name", 'BIAS_Ranking',
                                                'COMPLEXITY_Ranking',
                                                'EFFECTIVENESS_Ranking',
                                                'EXPENSIVENESS_Ranking',
                                                'EXPLAINABLE_Ranking',
                                                'SAFE_Ranking',
                                                'TRUSTWORTHY_Ranking'
                        )
    )

elif add_selectbox == "Reasoning-v1.5":

    st.dataframe(
        dfv2.style.background_gradient(subset=['explainable_score', 'trustworthy_score'],cmap='YlOrRd'),
        use_container_width= True,
        height = 700,
        hide_index=True,
        column_order= ("Use Case ID", "Use Case Name", "explainable_score", "trustworthy_score"
                        )

    )

elif add_selectbox == "cosine-similarity-v1.5":
    st.dataframe(
        dfcosv2.style.background_gradient(subset=['cosine_explainable_score', 'cosine_trustworthy_score'],cmap='YlOrRd'),
        use_container_width= True,
        height = 700,
        hide_index=True,
        column_order= ("Use Case ID", "Use Case Name", "cosine_explainable_score", "cosine_trustworthy_score"
                        )

    )



