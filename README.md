# Quick Note
> [!NOTE]
> This project is mainly for my own learning and experimentation. It is not intended for production use.
> it was made with the intention of being a simple experiment to learn how to build a RAG. if you find it useful
> and consider this has potential to be useful to you, please consider contributing to the project.
> and making sugestions for improvements.

# Doc Scraper AI

Doc Scraper AI is a Streamlit-based web application for scraping, processing, and querying documentation from web sources. It leverages modern NLP and vector database technologies to enable semantic search and Q&A over scraped documentation.

## Features
- **Streamlit Web App**: User-friendly interface for interacting with the application.
- **Documentation Scraping**: Extracts and cleans documentation from specified URLs.
- **Embeddings & Vector Search**: Uses Sentence Transformers and ChromaDB for semantic search.
- **LLM Integration**: Supports querying with Google Gemini and Ollama (Qwen3) models.
- **Dockerized Deployment**: Easily run the app and ChromaDB server using Docker Compose.

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.12 (for local development)

### Quick Start (Docker)
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/doc-scraper-ai.git
   cd doc-scraper-ai
   ```
2. Create a `.env` file with the following variables:
   ```env
   STREAMLIT_PORT=8501
   CHROMA_PORT=8000
   ```
3. Start the application:
   ```bash
   docker-compose up --build
   ```
4. Access the Streamlit app at `http://localhost:8501`.

### Local Development
1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Project Structure
- `app.py` — Main Streamlit application.
- `url-code-extractor.py` — Utility for extracting code from URLs.
- `docs_url.pickle` — Stores scraped documentation data.
- `data/chroma/` — Persistent storage for ChromaDB.
- `Dockerfile` & `docker-compose.yml` — Containerization and orchestration.
- `generate-embeddings.ipynb`, `url-doc-extract.ipynb` — Jupyter notebooks for prototyping and testing (not required for main app).

## Usage
1. Scrape documentation using the app or provided scripts.
2. Generate embeddings and store them in ChromaDB.
3. Query documentation using natural language via the Streamlit interface.

## LLM Integration
- **Google Gemini**: Requires API key in `Gemini.key` file.
- **Ollama (Qwen3)**: Requires local Ollama server running on port 11434.

## Contributing
Pull requests and issues are welcome! Please open an issue to discuss major changes.

## License
MIT License

## Acknowledgements
- [Streamlit](https://streamlit.io/)
- [ChromaDB](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain](https://www.langchain.com/)
- [Google Gemini](https://ai.google.dev/)
- [Ollama](https://ollama.com/)
