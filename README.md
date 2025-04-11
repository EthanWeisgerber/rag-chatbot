# RAG Chatbot: Document-Aware AI Assistant

This project is a Retrieval-Augmented Generation (RAG) chatbot that uses OpenAI's GPT-3.5, LangChain, and Pinecone to answer user questions based on the content of uploaded documents. It supports `.pdf`, `.txt`, and `.csv` files and can be used interactively in the terminal or deployed as a Gradio web app.

## Key Features

- Semantic search using Pinecone vector database
- Supports multi-file ingestion (.pdf, .txt, .csv)
- GPT-3.5-based response generation with contextual awareness
- Automatic text chunking for long documents
- Interactive terminal-based chat
- Optional Gradio web interface for easier access

## Example Use Case

The app was tested with a **1 Timothy Bible commentary** and successfully used during a Bible study session to answer scripture-related questions with relevant excerpts.

## How It Works

1. **Load documents** from local files.
2. **Split** them into overlapping chunks for optimal context handling.
3. **Embed** the chunks using OpenAI Embeddings.
4. **Store** the embeddings in a Pinecone vector index.
5. **Query** Pinecone using the user’s input.
6. **Generate responses** using GPT-3.5, incorporating top retrieved context chunks.

## Technologies Used

- **Python**
- **LangChain**
- **OpenAI GPT-3.5 & Embeddings**
- **Pinecone**
- **Gradio (optional)**
- **dotenv** for environment variable management

## File Support

- `.pdf`: Loaded via `PyPDFLoader`
- `.csv`: Loaded via `CSVLoader`
- `.txt`: Loaded as plain text

## Running the Chatbot (Terminal)

1. Clone the repo.
2. Add your OpenAI and Pinecone API keys to a `.env` file:

   ```env
   OPENAI_API_KEY=your_openai_key
   PINECONE_API_KEY=your_pinecone_key
   ```
3. Run the script:

   ```env
   python chatbot_app.py
   ```
4. Follow the prompts to add documents and ask questions.

## Sample Interaction

```text
User: What does the 1 Timothy commentary say about elders?

Chatbot: "According to the provided commentary, elders are to be above reproach..."
```

## Notes

- The system uses cosine similarity for semantic retrieval.
- Embeddings are stored by chunk with source and page metadata.
- Responses are grounded in uploaded content, with fallback to model knowledge if needed.

## Acknowledgments

- OpenAI for GPT and embedding APIs
- Pinecone for vector storage
- LangChain for chaining LLMs with document processing
