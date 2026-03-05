import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """Load all text file from the docs dir"""
    print(f"Loading the documents form {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exists. Please crerate it.")
    
    loader = DirectoryLoader(
        path=docs_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt file found in {docs_path}. Please add the documents.")
    
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}...")
        print(f" Metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
    """Split the documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        )
    
    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunks {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5 :
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store"""
    print("Creating embedding and storing in ChromaDB...")

    embedding_model = MistralAIEmbeddings(model="mistral-embed")

    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished crerating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    print("Main Function")

    documents = load_documents(docs_path="docs")

    chunks = split_documents(documents)

    vectorstore = create_vector_store(chunks)



if __name__ == "__main__":
    main()