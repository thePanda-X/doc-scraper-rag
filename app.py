
import os
from urllib.parse import urljoin, urlparse, urlunparse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import streamlit as st
import chromadb

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# Config from .env
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docs")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8081"))

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize ChromaDB HTTP client
try:
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    chroma_client.heartbeat()
    st.success(f"Successfully connected to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}!")
except Exception as e:
    st.error(f"Failed to connect to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}. Please ensure your Docker container is running and accessible. Error: {e}")
    st.stop()

# Pass the client to the Chroma LangChain wrapper
vectordb = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model,
)

# Initialize retriever and LLM chain
retriever = vectordb.as_retriever(search_kwargs={"k": 5})

ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
llm = OllamaLLM(model="qwen3:0.6b", reasoning=False, host=ollama_host)

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Provide examples and direct quotes from the context when possible."
    "Context: {context}"
)

# Create a PromptTemplate instance
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Initialize the QA chain with the custom prompt
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

def is_valid_url(url, base_url):
    """
    Check if `url` is under the `base_url` domain and path.
    """
    parsed = urlparse(url)
    base_parsed = urlparse(base_url)

    if parsed.scheme != base_parsed.scheme:
        return False

    if parsed.netloc != base_parsed.netloc:
        return False

    base_path = base_parsed.path
    if not base_path.endswith("/"):
        base_path += "/"

    url_path = parsed.path
    if not url_path.endswith("/"):
        url_path += "/"

    return url_path.startswith(base_path)

    

def extract_links(soup, base_url):
    """
    Helper method to extract all valid links from the BeautifulSoup object.
    """
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(base_url, href)
        if is_valid_url(full_url, base_url):
            links.add(full_url)
    return links


def get_html_content(soup):
    parts = []
    for tag in soup.find_all(["h1", "h2", "h3", "p", "pre", "code", "li"]):
        if tag.name in ["h1", "h2", "h3"]:
            parts.append(f"# {tag.get_text(strip=True)}")
        elif tag.name == "p":
            parts.append(tag.get_text(strip=True))
        elif tag.name in ["pre", "code"]:
            lang = ""
            if tag.has_attr('class'):
                for cls in tag['class']:
                    if cls.startswith('language-'):
                        lang = cls.split('-')[1]
                        break
            parts.append(f"```{lang}\n{tag.get_text(strip=True)}\n```")
        elif tag.name == "li":
            parts.append(f"- {tag.get_text(strip=True)}")
    return "\n\n".join([p for p in parts if p.strip()])


def normalize_url(url):
    """Normalize URL by removing fragment and query, ensuring consistent format."""
    parsed = urlparse(url)
    normalized = parsed._replace(query="", fragment="")
    return urlunparse(normalized).rstrip("/")


def crawl_and_scrape(base_url, max_pages=250, progress_bar_placeholder=None, progress_text_placeholder=None):
    """
    Crawl the site starting from the base URL and extract the HTML content of each page.
    Includes Streamlit progress updates.
    """
    visited = set()
    to_visit = {normalize_url(base_url)}
    all_content = dict()

    # Initial update for the progress bar
    if progress_bar_placeholder and progress_text_placeholder:
        progress_bar_placeholder.progress(0)
        progress_text_placeholder.text(f"Starting crawl for {base_url} (0/{max_pages} pages scraped)...")

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop()
        if current_url in visited:
            continue

        # Update progress before processing the current URL
        if progress_bar_placeholder and progress_text_placeholder:
            current_progress = len(visited) / max_pages
            progress_bar_placeholder.progress(current_progress)
            progress_text_placeholder.text(f"Crawling: {current_url} ({len(visited)}/{max_pages} pages scraped)")
        
        print(f"{len(visited)}/{max_pages} Crawling: {current_url}") # For console logging
        visited.add(current_url)

        try:
            response = requests.get(current_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            links = {
                normalize_url(link)
                for link in extract_links(soup, base_url)
            }
            new_links = links - visited
            to_visit.update(link for link in new_links if len(visited) + len(to_visit) < max_pages)

            html_content = get_html_content(soup)
            if html_content.strip():
                all_content[current_url] = html_content

        except requests.exceptions.Timeout:
            print(f"Timeout occurred while fetching URL: {current_url}")
        except requests.RequestException as e:
            print(f"An error occurred while fetching {current_url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred with {current_url}: {e}")

    # Final update after loop completes
    if progress_bar_placeholder and progress_text_placeholder:
        progress_bar_placeholder.progress(1.0) # Ensure it reaches 100%
        progress_text_placeholder.success(f"Finished crawling {len(visited)} pages for {base_url}.")

    return all_content


def add_docs_to_chroma(pages_dict):
    """
    Adds scraped content from pages_dict to ChromaDB in batches to respect the max_batch_size,
    with improved visualization for embedding generation.
    """
    all_texts = []
    all_metadatas = []

    for url, content in pages_dict.items():
        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        
        all_texts.extend(chunks)
        all_metadatas.extend([{"source": url}] * len(chunks))
    
    if all_texts:
        total_chunks = len(all_texts)
        
        embedding_status_text = st.empty()
        embedding_progress_bar = st.progress(0)
        
        embedding_status_text.info(f"Starting embedding generation and adding {total_chunks} chunks to Chroma DB...")

        try:
            max_batch_size = vectordb._client.get_max_batch_size()
        except Exception as e:
            st.error(f"Could not retrieve max_batch_size from ChromaDB client. Defaulting to 1000. Error: {e}")
            max_batch_size = 1000

        if max_batch_size == 0:
            max_batch_size = 1000

        num_batches = (total_chunks + max_batch_size - 1) // max_batch_size
        
        for i in range(0, total_chunks, max_batch_size):
            batch_texts = all_texts[i : i + max_batch_size]
            batch_metadatas = all_metadatas[i : i + max_batch_size]
            
            current_batch_num = i // max_batch_size + 1
            chunks_processed = i + len(batch_texts)

            embedding_status_text.info(f"Generating embeddings and adding batch {current_batch_num}/{num_batches} ({chunks_processed}/{total_chunks} chunks processed)...")
            
            try:
                vectordb.add_texts(texts=batch_texts, metadatas=batch_metadatas)
                embedding_progress_bar.progress(chunks_processed / total_chunks)
            except Exception as e:
                st.error(f"Failed to add batch starting at index {i}. Error: {e}")
                embedding_status_text.error("Embedding and adding process failed for a batch. Check logs for details.")
                break

        embedding_status_text.success(f"All {total_chunks} chunks embedded and added to Chroma vector store!")
        embedding_progress_bar.progress(1.0)
    else:
        st.warning("No content found to add to Chroma DB.")

def chroma_db_has_data():
    """
    Checks if the specified ChromaDB collection contains any documents.
    Updates st.session_state.last_embedded_count and st.session_state.docs_embedded.
    """
    try:
        collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME) 
        
        count = collection.count()
        
        st.session_state.last_embedded_count = count
        if count > 0:
            st.session_state.docs_embedded = True
            return True
        else:
            st.session_state.docs_embedded = False
            return False
    except Exception as e:
        st.error(f"Error checking ChromaDB collection '{COLLECTION_NAME}': {e}. This might mean the collection doesn't exist yet, or ChromaDB is unreachable.")
        st.session_state.docs_embedded = False
        st.session_state.last_embedded_count = 0
        return False

chroma_db_has_data() # Initialize db state variables

st.set_page_config(layout="wide", page_title="RAG Doc Scraper + Chatbot")

st.title("📚 RAG Doc Scraper + Chatbot")

# Initialize session state variables for persistent data and flags
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

tab1, tab2 = st.tabs(["Scrape & Embed Docs", "Chat with Docs"])

with tab1:
    st.header("🔍 Scrape Documentation & Build Knowledge Base")
    st.markdown("""
    Enter the URLs of the documentation websites you want to scrape. 
    The scraper will crawl these sites, extract content, and then embed it into your local ChromaDB
    to create a knowledge base for the chatbot.
    """)

    urls_input = st.text_area(
        "Enter URLs (one per line, e.g., `https://example.com/docs/`):",
        value="https://fastapi.tiangolo.com/tutorial/\nhttps://fastapi.tiangolo.com/reference/",
        height=120,
        placeholder="https://your-doc-site.com/docs/\nhttps://another-site.org/wiki/"
    )
    max_pages = st.number_input(
        "Max pages to scrape per initial URL (to prevent excessive crawling):",
        min_value=1, max_value=250, value=3, step=1,
        help="Sets a limit on how many unique pages the scraper will visit starting from each provided URL."
    )
    
    col1, col2 = st.columns([1, 2])
    with col1:
        scrape_button = st.button("🚀 Start Scraping and Embedding")
    with col2:
        if st.session_state.docs_embedded:
            st.success(f"✅ Currently, {st.session_state.last_embedded_count} documents are embedded and ready for chat.")
        else:
            st.info("No documents embedded yet. Start scraping to build your knowledge base!")

    st.divider() # Visual separator

    if scrape_button:
        urls = [u.strip() for u in urls_input.split("\n") if u.strip()]
        if not urls:
            st.error("🚨 Please enter at least one URL to scrape.")
        else:
            st.info("Starting the scraping and embedding process. This may take a while...")
            all_scraped_content = {}
            total_initial_urls = len(urls)
            
            # Overall progress for initial URLs
            overall_url_progress_text = st.empty()
            overall_url_progress_bar = st.progress(0)

            # Containers for detailed per-URL progress
            scrape_details_container = st.container()

            for i, url in enumerate(urls):
                current_url_display = f"Processing initial URL {i+1}/{total_initial_urls}: **`{url}`**"
                overall_url_progress_text.markdown(current_url_display)
                
                with scrape_details_container.expander(f"Details for {url}", expanded=True):
                    st.subheader(f"🌐 Crawling pages for: `{url}`")
                    current_scrape_progress_text = st.empty()
                    current_scrape_progress_bar = st.progress(0)

                    # Replace with your actual crawl_and_scrape function
                    scraped_pages = crawl_and_scrape(
                        url,
                        max_pages=max_pages,
                        progress_bar_placeholder=current_scrape_progress_bar,
                        progress_text_placeholder=current_scrape_progress_text
                    )
                    all_scraped_content.update(scraped_pages)
                
                overall_url_progress_bar.progress((i + 1) / total_initial_urls)
            
            overall_url_progress_text.success(
                f"🎉 Finished crawling {total_initial_urls} initial URLs. "
                f"Collected a total of **{len(all_scraped_content)}** unique pages for embedding."
            )
            overall_url_progress_bar.progress(1.0) # Ensure it's full

            st.divider() # Visual separator

            if all_scraped_content:
                st.subheader("📊 Embedding Scraped Content into ChromaDB")
                embedding_status_text = st.empty()
                embedding_progress_bar = st.progress(0)
                
                try:
                    # Replace with your actual add_docs_to_chroma function
                    add_docs_to_chroma(
                        all_scraped_content,
                        embedding_progress_bar,
                        embedding_status_text
                    )
                    st.success(f"✅ Embedding complete! Your knowledge base is updated with {len(all_scraped_content)} documents.")
                    st.session_state.docs_embedded = True
                    st.session_state.last_embedded_count = len(all_scraped_content)
                    
                    st.balloons() # Little celebration
                except Exception as e:
                    st.error(f"❌ Error during embedding: {e}. Please ensure ChromaDB is running and accessible.")
                    embedding_status_text.error("Embedding failed.")
            else:
                st.warning("⚠️ No content was scraped. Nothing to embed.")
                st.session_state.docs_embedded = False
                st.session_state.last_embedded_count = 0


with tab2:
    st.header("💬 Chat with your Documentation")
    st.markdown("""
    Ask questions about the documentation you just scraped and embedded. 
    The chatbot will use the custom knowledge base to provide relevant answers.
    """)

    # Check if any documents are embedded before allowing chat
    if not st.session_state.docs_embedded:
        st.warning("ℹ️ No documents have been scraped and embedded yet. "
                   "Please go to the 'Scrape & Embed Docs' tab first.")
    else:
        st.info(f"Ready to chat! Your knowledge base contains approximately "
                f"{st.session_state.last_embedded_count} documents.")

        # Create a container for chat messages with a fixed height and scrollbar
        chat_history_container = st.container(height=500, border=True) 
        
        with chat_history_container:
            # Display chat history within the scrollable container
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
        
        # Place the chat input *outside* the history container, at the bottom
        if prompt := st.chat_input("Ask a question about the docs..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Immediately display user message in the history container
            with chat_history_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = chain.invoke({"input": prompt})
                        answer = response["answer"]

                        st.markdown(answer)
                    except Exception as e:
                        st.error(f"❌ Error communicating with LLM or ChromaDB: {e}. Ensure Ollama and ChromaDB containers are running correctly.")
                        answer = "I'm sorry, I couldn't process your request due to an internal error. Please check the console for details."
            
            st.session_state.chat_history.append({"role": "assistant", "content": answer})