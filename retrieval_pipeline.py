
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = MistralAIEmbeddings(model="mistral-embed")

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

query = "How much did Microsoft pay to acquire GitHub?"

retriever = db.as_retriever(search_kwargs={"k": 5})

#retriever = db.as_retriever(
#    search_type="similarity_score_threshold",
#    search_kwargs={
#        "k":5,
#        "score_threshold": 0.3
#    }
#)

relevent_docs = retriever.invoke(query)

print(f"User Query: {query}")

print("--- Context ---")
for i, doc in enumerate(relevent_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

combined_input = f"""Based on the following documents, please answer this queston: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevent_docs])}

Please provide a clear, helpful answer using only the information from the documents. If you can't find the answer in the document, say you don't know.
"""

model = ChatMistralAI(model="mistral-large-latest")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)

print("\n--- Generate Response ---")
print("Content only:")
print(result.content)

# In which year did Tesla begin production of the Roadster?