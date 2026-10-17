###Data injestion
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader("Python_Generative_AI_Beginner_Guide.pdf")
text=loader.load()

###chunking that is data transformation 

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
docs=text_splitter.split_documents(text)
print(docs)

### Embeddings convert text to vectors
from dotenv import load_dotenv
import os

load_dotenv()

# print("API key loaded:", bool(os.getenv("GOOGLE_API_KE    Y")))
    
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

# embeddings = GoogleGenerativeAIEmbeddings(
#     model="gemini-embedding-001"
# )
# print(embeddings)

embeddings= OllamaEmbeddings(
    model="nomic-embed-text"
)
#vector store
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    docs,
    embeddings
)

print("FAISS vector store created!")
    # # -------------------------
# # 5. Create retriever
# # -------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# # -------------------------
# # 6. LLM
# # -------------------------
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# # -------------------------
# # 7. Prompt
# # -------------------------
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using only the context.

    Context:
    {context}

    Question:
    {question}

    If the answer is not available,
    say that you don't know.
    """
)


# # -------------------------
# # 8. Ask question
# # -------------------------

question = "What is RAG?"

docs = retriever.invoke(question)


context = "\n\n".join(
    doc.page_content
    for doc in docs
)


# # -------------------------
# # 9. Generate answer
# # -------------------------
chain = prompt | llm

response = chain.invoke({
    "context": context,
    "question": question
})

print(response.content)