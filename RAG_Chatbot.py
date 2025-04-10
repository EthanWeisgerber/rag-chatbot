from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
# UnstructuredHTMLLoader
from pinecone import Pinecone, ServerlessSpec
import os
import time
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from langchain.schema import Document

load_dotenv()

# Ask for the Pinecone index name at the beginning
index_name = input("Enter the index name (new or already created): ").strip().lower()

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)

# Initialize Embedding & Chat Model
model = 'gpt-3.5-turbo'
embed_model_type = 'text-embedding-ada-002'
embed_model = OpenAIEmbeddings(model=embed_model_type)
chat = ChatOpenAI(model=model)


def create_pinecone_index(index_name):
    """
    Creates a Pinecone index with the specified name if it doesn't already exist.
    Args:
        index_name (str): The name of the Pinecone index to create or connect to.
    Returns:
        pinecone.Index: A handle to the Pinecone index.
    """
    existing_indexes = [index['name'] for index in pc.list_indexes().get('indexes', [])]
    if index_name not in existing_indexes:
        
        print(f"Creating Pinecone index: {index_name}...")
        pc.create_index(
            name=index_name,
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
            dimension=1536,
            metric="cosine"
        )
        
        # Wait for index creation
        timeout = 60
        start_time = time.time()
        while time.time() - start_time < timeout:
            status = pc.describe_index(index_name).status
            if status['ready']:
                print(f"Index {index_name} is ready!")
                break
            time.sleep(1)
    else:
        print(f"{index_name} index found!")
    return pc.Index(index_name)

# Create index (if not already created)
index = create_pinecone_index(index_name)

def load_files(file_names):
    """
    Loads and parses documents from a list of file names. Supports .txt, .pdf, and .csv.
    Args:
        file_names (list[str]): A list of file paths to load.
    Returns:
        dict[str, list[Document]]: A dictionary mapping file names to lists of LangChain Documents.
    """
    file_data = {}

    for file_name in file_names:
        if file_name.endswith(".txt"):
            with open(file_name, "r", encoding="utf-8") as file:
                content = file.read()
                documents = [Document(page_content=content)]
        elif file_name.endswith(".pdf"):
            documents = PyPDFLoader(file_name).load()
        elif file_name.endswith(".csv"):
            documents = CSVLoader(file_name).load()
        else:
            print(f"File type not supported: {file_name}")
            continue

        file_data[file_name] = documents  # Store processed documents

    return file_data

def split_docs(file_data, chunk_size=500, chunk_overlap=50):
    """
    Splits document content into overlapping text chunks for embedding.
    Args:
        file_data (dict[str, list[Document]]): The dictionary of file documents to split.
        chunk_size (int): Maximum number of characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
    Returns:
        dict[str, list[Document]]: A dictionary mapping file names to lists of chunked Documents.
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunked_docs = {}
    for file_name, documents in file_data.items():
        chunked_docs[file_name] = splitter.split_documents(documents)
    
    return chunked_docs

def embed_docs(chunked_docs):
    """
    Embeds text chunks and uploads them to the Pinecone index in batches.
    Args:
        chunked_docs (dict[str, list[Document]]): Dictionary of chunked documents to embed and upsert.
    Returns:
        None
    """
    for file_name, chunked_document in chunked_docs.items():
        texts = [chunk.page_content for chunk in chunked_document]
        embeddings = embed_model.embed_documents(texts)

        to_upsert = [
            (
                f"{file_name}-chunk-{i}",  # Unique ID with filename
                embeddings[i],  # Corresponding embedding vector
                {
                    "text": chunked_document[i].page_content,
                    "page": chunked_document[i].metadata.get("page", "Unknown"),
                    "source": file_name
                }
            )
            for i in range(len(chunked_document))
        ]

        for i in range(0, len(to_upsert), 100):  # Batch upserts
            batch = to_upsert[i : i + 100]
            index.upsert(vectors=batch)

def chatbot(user_query):
    """
    Responds to a user query using relevant document chunks retrieved from the Pinecone index.
    Args:
        user_query (str): The question posed by the user.
    Returns:
        str: The chatbot's generated response based on context and the LLM's reasoning.
    """
    query_embedding = embed_model.embed_query(user_query)

    # Query the single index
    result = index.query(vector=query_embedding, top_k=10, include_metadata=True)
    retrieved_chunks = [match["metadata"]["text"] for match in result["matches"]]

    context = "\n\n".join(retrieved_chunks)

    messages = [
        SystemMessage(content="You are a helpful assistant trained on multiple documents. Use the provided excerpts to answer the user's question."),
        HumanMessage(content=f"### Context from the provided documents:\n{context}\n\n### User's Question:\n{user_query} If the answer cannot be found in the provided context, let the user know. You can then provide an answer based off your knowledge, but make sure you let user know that it is not from the documents provided. Also remind them that they can type 'exit' to exit the chat!\n\n### Answer:")
    ]

    response = chat.invoke(messages)
    return response.content

# Interactive Menu for Adding Files & Chatting
print("\nWelcome to the RAG Chatbot!")

while True:
    action = input("\nOptions:\n1️⃣ Add Files\n2️⃣ Start Chat\n3️⃣ Exit\nChoose an option (1/2/3): ")

    if action == "1":  # Add multiple files
        command = ''
        file_list = []
        while command != 'done':
            file = input("Enter a file name or type 'done' to move on: ").strip()
            if file.lower() == "done":
                break
            elif not file.endswith(('.pdf', '.txt', '.csv', '.html')):
                print("Unsupported file type.")
            else:
                file_list.append(file)
        if file_list:
            file_list = [file.strip() for file in file_list]
            file_data = load_files(file_list)  # Load multiple files
            chunked_docs = split_docs(file_data)  # Chunk documents
            embed_docs(chunked_docs)  # Embed and store in Pinecone
            print("Files have been successfully added to the index!")
        else:
            print("No valid files were added.")

    elif action == "2":  # Start chat
        print("\nChatbot is ready! Type 'exit' to quit.")
        while True:
            user_query = input("Ask the chatbot: ")
            if user_query.lower() == "exit":
                break
            response = chatbot(user_query)
            print("\nChatbot:", response)

    elif action == "3":  # Exit
        print("Goodbye!")
        break
