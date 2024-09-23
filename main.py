import os
import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent

# Set the API Key for Groq (ensure to replace with your actual key)
os.environ["GROQ_API_KEY"] = st.secrets["groq_api_key"]  # Assuming you use Streamlit secrets management

# Initialize the Groq LLM
llm = ChatGroq(
    temperature=0,
    model_name="mixtral-8x7b-32768"  # Replace with your desired model
)

# Streamlit app title
st.title("CSV Query Agent with Groq")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the CSV content as a dataframe
    df = pd.read_csv(uploaded_file)
    st.write("CSV Data Preview:")
    st.dataframe(df.head())  # Display the first few rows of the CSV file

    # Create the CSV agent
    csv_agent = create_csv_agent(
        llm,
        uploaded_file,  # Use the uploaded file
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True  # Enable for advanced code execution
    )

    # Input for user query
    user_query = st.text_input("Enter your query:")

    if st.button("Submit"):
        if user_query:
            # Example query to interact with the CSV
            response = csv_agent(user_query)
            st.write("Response:")
            st.write(response)
        else:
            st.warning("Please enter a query to submit.")
