from fastapi import FastAPI
import pandas as pd

from search_engine import search_vessels
from llm_interface import ask_llm

app = FastAPI()

df = pd.read_csv("data/sample_vessels.csv")


@app.get("/query")

def query_system(q:str):

    results = search_vessels(df,q)

    answer = ask_llm(q,results)

    return {
        "results":results,
        "llm_answer":answer
    }