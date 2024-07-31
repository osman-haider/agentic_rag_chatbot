import streamlit as st
import os
import pandas as pd
from pandasai import SmartDataframe
import matplotlib.pyplot as plt
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

image_path = 'exports/charts/temp_chart.png'

def upload_csv():
    st.session_state.prompt_history = []
    st.session_state.openai_key = openai_api_key
    st.sidebar.title("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        # Accept user input
        if prompt := st.chat_input("What is up?"):

            llm = OpenAI(api_token=st.session_state.openai_key)
            pandas_ai = SmartDataframe(st.session_state.df, config={"llm": llm})
            response = pandas_ai.chat(prompt)

            if os.path.exists(image_path):
                print("file is present")
                im = plt.imread(image_path)
                st.image(im)
                os.remove(image_path)

            elif response is not None:
                st.write(response)


    else:
        st.write("Please upload a CSV file.")

# Main function to render the Streamlit app
def main():
    st.title("CSV File Upload App")
    upload_csv()


if __name__ == "__main__":
    main()
