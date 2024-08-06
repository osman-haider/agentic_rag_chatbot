import streamlit as st
import os
import pandas as pd
from pandasai import SmartDataframe
import matplotlib.pyplot as plt
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai


# Load environment variables from .env file
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
image_path = 'exports/charts/temp_chart.png'

os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_conversational_chain():
    prompt_template = """
    You are an AI chatbot designed to enhance responses based on the information available in a CSV file. When providing answers, you should enrich them as much as possible. If a specific answer is not available, please use a personalized apology such as: "I am sorry, I currently do not have information about [specific topic]."

    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question, df):
    llm = OpenAI(api_token=st.session_state.openai_key)
    pandas_ai = SmartDataframe(df, config={"llm": llm})
    response = pandas_ai.chat(user_question)
    print(type(response))
    # Ensure the response is in the correct format
    if isinstance(response, int):
        st.error("Error: The response from pandas_ai.chat is an integer, expected a list or string.")
        return "Sorry, something went wrong."

    # Ensure response is a list of documents or strings
    if not isinstance(response, (list, str)):
        st.error("Error: The response from pandas_ai.chat is not a list or string.")
        return "Sorry, something went wrong."

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": response, "question": user_question},
        return_only_outputs=True
    )
    return response


def login_screen():
    st.title("Welcome to Data Agentic Hub")
    st.subheader("Your Agentic Data Analyst Assistant")
    st.write("""
        **Connect, Analyze, Visualize, and Present Your Data Effortlessly**
        1. Connect your CSV file and ask any data-related question.
        2. Create insightful graphs, pie charts, and more.
        3. Summarize datasets and generate detailed PowerPoint presentations.
        ============================ Login to Your Account ============================
    """)
    username = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "123":  # Simple hardcoded check
            st.session_state.logged_in = True
        else:
            st.error("Invalid username or password")
def upload_csv():
    st.session_state.prompt_history = []
    st.session_state.openai_key = openai_api_key
    st.title("Welcome Back!")
    st.write("""
        1. Interact with your data: ask questions, visualize trends, and more.
        2. Generate reports and insights with just a few clicks.
        **Get started by uploading your file and let’s unlock the power of your data!**
    """)
    st.sidebar.title("Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        # Accept user input
        if prompt := st.chat_input("What is up?"):
            response = user_input(prompt, st.session_state.df)
            # llm = OpenAI(api_token=st.session_state.openai_key)
            # pandas_ai = SmartDataframe(st.session_state.df, config={"llm": llm})
            if "graph" in prompt.lower():
                # Handle graph requests
                # response = pandas_ai.chat(prompt)
                if os.path.exists(image_path):
                    # print("file is present")
                    im = plt.imread(image_path)
                    st.image(im)
                    # st.pyplot(im)
                    # os.remove(image_path)
                elif response is not None:
                    st.write(response)
            else:
                # Handle description requests
                # response = pandas_ai.chat(prompt)
                st.write(response)
    else:
        st.title("Please upload a CSV file.")
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if not st.session_state.logged_in:
        login_screen()
    else:
        # st.title("CSV File Upload App")
        upload_csv()
if __name__ == "__main__":
    main()