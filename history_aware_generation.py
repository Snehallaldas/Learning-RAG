from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

load_dotenv()

persistent_directory = "db/chroma_db"
embeddings = MistralAIEmbeddings(model="mistral-embed")
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

model = ChatMistralAI(model="mistrtal-large-latest")

chat_history = []

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    if chat_history:
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question without any additional text.")
            ] + chat_history + [
                HumanMessage(content=f"New question: {user_question}")
            ]
    else:
        search_question = user_question

    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevent documents:")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f" Doc {i}: {preview}...")

    cobined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from the documents. If you can't find the answer in the document, say you don't know.
    """

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=cobined_input)
    ]
    result = model.invoke(messages)
    answer = result.content

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(f"Answer: {answer}")
    return answer

def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")
        if question.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break