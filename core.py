from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from langchain.chains import RetrievalQA  # Not needed with LCEL

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_medical_answer(user_query):
    #Load the memory
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(
        persist_directory="./medical_db", 
        embedding_function=embedding
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    
    #Initialize the Ollama which is running locally 
    llm = ChatOllama(model ="llama3.1:8b", temperature=0)

    #Creating a medical specific prompt template
    template = """You are a specialized medical research assistant. 
    Your expertise is strictly limited to the provided research context.

    STRICT GUARDRAILS:
    1. Answer ONLY using the information from the Context below.
    2. If the Context does not contain the answer, you MUST say: "I cannot find information about this in the current medical database."
    3. Do NOT use your internal training data to answer general knowledge questions (e.g., capitals, history, celebrities).
    4. Do NOT perform creative writing tasks (e.g., poems, stories, jokes).
    5. If the question is about medical advice not found in the context, decline to answer.

    Context:
    {context}

    Question: {question}

    Answer (based strictly on context):"""

    prompt = ChatPromptTemplate.from_template(template)

    #The chain 
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    #Run it 
    docs = retriever.invoke(user_query)
    answer = rag_chain.invoke(user_query)

    return answer, docs
