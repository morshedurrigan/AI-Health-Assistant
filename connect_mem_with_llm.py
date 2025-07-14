import os

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Step 1: Setup LLM (Mistral with HuggingFace)
HF_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
        Your goal is to provide accurate, supportive, and professional responses based on the given context. 

        - Use only the information provided in the context to answer the user's question.
        - If the answer is not available in the context, kindly say, "I'm sorry, but I don't have that information."
        - Respond warmly and naturally to greetings like "hi" or "hello."
        - Maintain a compassionate, reassuring, and professional tone in all responses.
        - Keep answers concise yet informative, avoiding unnecessary details.
        - Do not mention whether context is available—just provide a clear, direct, and helpful response.
        - Focus solely on the current question without referencing previous interactions unless the user explicitly asks.

        Context: {context}  
        Question: {question}  

        Provide a thoughtful and accurate response while ensuring empathy and clarity.
        """  

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"  # Make sure this path exists and has the correct FAISS files
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Create QA chain using the locally loaded LLM
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),  # Use the Mistral LLM loaded from HuggingFace
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=False,
    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
)

# Now invoke with a single query
user_query = input("Write Query Here: ")
response = qa_chain.run(user_query)  # Correct method is .run(), not .invoke()
print("RESULT: ", response)  # Directly print the response
