# backend.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline

# -----------------------------
# Load knowledge base
# -----------------------------
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,       # reduced from 800 to avoid sequence length issues
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)

# -----------------------------
# Embeddings (fast & free)
# -----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# -----------------------------
# Vector Store (persistent & cached)
# -----------------------------
persist_dir = "db"

if os.path.exists(persist_dir):
    # Load existing vector store (cached embeddings)
    vector_store = Chroma(
        collection_name="consumer_protection_kb",
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )
else:
    # Create new vector store and persist
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="consumer_protection_kb",
        persist_directory=persist_dir
    )
    vector_store.persist()

# Retriever for RAG
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# -----------------------------
# Prompt Template
# -----------------------------
prompt_template = """
You are a helpful assistant that answers questions based on the provided context.
You must only use the information from the context to answer the question. Do not use any prior knowledge.
If the answer is not contained in the context, say that you cannot answer.

Context:
{context}

Question:
{question}

Answer:
"""
PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# -----------------------------
# Free Local LLM (t5-small)
# -----------------------------
generator = pipeline(
    "text2text-generation",
    model="t5-small",
    device_map="auto",  # uses MPS on M1 Air
    max_new_tokens=150
)
llm = HuggingFacePipeline(pipeline=generator)

# -----------------------------
# RAG Chain
# -----------------------------
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

# -----------------------------
# Function to answer queries
# -----------------------------
def get_answer(query: str) -> str:
    """Answer user queries using the RAG pipeline."""
    return rag_chain.invoke(query)

# -----------------------------
# Optional: test CLI
# -----------------------------
if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is a consumer dispute?"
    print(f"\n🔎 Query: {query}\n")

    # Show top retrieved doc (first 500 chars only)
    retrieved_docs = vector_store.similarity_search(query, k=1)
    print("📄 Retrieved Document:")
    if retrieved_docs:
        print(f"\n--- Doc 1 ---\n{retrieved_docs[0].page_content[:500]}...\n")

    # Generate final answer
    answer = get_answer(query)
    print("💡 Answer:\n")
    print(answer)
