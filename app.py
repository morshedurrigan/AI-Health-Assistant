import os
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

DB_FAISS_PATH = "vectorstore/db_faiss"

def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt():
    return PromptTemplate(
        template = """
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
        """,
        input_variables=["context", "question"]
    )

def load_llm():
    # Create the endpoint for conversational task
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        task="conversational",  # ← Specify conversational task
        huggingfacehub_api_token=os.environ.get("HF_TOKEN"),
        max_new_tokens=512,
        temperature=0.5,
    )
    
    # Wrap with ChatHuggingFace for proper conversational interface
    return ChatHuggingFace(llm=llm)

def chatbot(prompt, history):
    try:
        vectorstore = get_vectorstore()
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=False,
            chain_type_kwargs={'prompt': set_custom_prompt()}
        )
        response = qa_chain.invoke({'query': prompt})
        return [{"role": "assistant", "content": response["result"]}]
    except Exception as e:
        return [{"role": "assistant", "content": f"Error: {str(e)}"}]

iface = gr.ChatInterface(
    fn=chatbot,
    title="AI Health Assistant",
    description="This chatbot provides supportive and professional responses to your health-related questions. Powered by LangChain, Hugging Face, and FAISS, it offers empathetic and accurate answers based on a curated knowledge base.",
    chatbot=gr.Chatbot(type="messages")
)

if __name__ == "__main__":
    iface.launch()