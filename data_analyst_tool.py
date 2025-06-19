# data_analyst_tool.py

import pandas as pd
import os
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

# --- Load Data and Configuration ---
try:
    df = pd.read_csv("housing_initial.csv")
    VALID_COLUMNS = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'MedHouseVal']
    df = df[VALID_COLUMNS]
    DF_HEAD_STR = str(df.head())
except FileNotFoundError:
    logger.error("Data file for analyst tool not found! Please run prepare_data.py.")
    df = pd.DataFrame()
    VALID_COLUMNS = []
    DF_HEAD_STR = "Error: Data file not found."


def synthesize_response_from_data(query: str, pandas_result: str) -> str:
    """Uses an LLM to turn raw data output into a concise, natural language response."""
    
    if len(pandas_result) > 3000:
        pandas_result = pandas_result[:3000] + "\n... (data truncated)"
        
    # ** THE CRUCIAL FIX FOR CONCISENESS IS IN THIS PROMPT **
    prompt = f"""
    You are a data analyst assistant. Your tone is professional, direct, and concise.
    Your job is to provide a clear, final answer to the user's question based on the provided data.

    **IMPORTANT CONTEXT & RULES:**
    - The 'MedHouseVal' column represents value in units of $100,000. You MUST multiply it by 100,000 for any final dollar amount and format it as currency (e.g., $250,000).
    - **DO NOT** explain your calculations (e.g., do not say "this is calculated by...").
    - **DO NOT** mention the original 'MedHouseVal' number. Only show the final, calculated dollar amount.
    - If the data is a table, summarize the key finding, don't just show the table.
    - Answer ONLY the user's question. Be direct.

    User's Original Question: "{query}"

    Raw Data Result:
    ---
    {pandas_result}
    ---

    Concise and Professional Response:
    """
    
    try:
        llm = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Error in response synthesis: {e}")
        return f"Here is the raw data I found:\n```\n{pandas_result}\n```"


def query_housing_data(query: str) -> str:
    """
    Answers a factual question about the housing dataset by generating, executing, and synthesizing the result.
    """
    prompt = f"""
    You are an expert Python Pandas programmer.
    Your task is to convert a natural language query into a single, executable line of Pandas code that operates on a dataframe named `df`.
    The dataframe `df` has the following columns: {VALID_COLUMNS}
    This is the head of the dataframe:
    {DF_HEAD_STR}

    RULES:
    1. The code MUST be a single line.
    2. You MUST only use the columns listed above. The price column is 'MedHouseVal'.
    3. Only output the raw Python code. Do not add explanations or markdown.

    Natural Language Query: "{query}"
    
    Pandas Code:
    """
    
    try:
        llm_coder = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = llm_coder.generate_content(prompt)
        pandas_code = response.text.strip().replace('`', '')

        logger.info(f"Generated Pandas Code: {pandas_code}")
        result = eval(pandas_code, {"df": df, "pd": pd})
        
        raw_result_str = str(result)
        logger.info(f"Execution result: {raw_result_str}")
        
        final_response = synthesize_response_from_data(query, raw_result_str)
        return final_response

    except Exception as e:
        logger.error(f"Error executing pandas code for query '{query}': {e}")
        return "I'm sorry, I encountered an error trying to analyze the data. Please rephrase your question."