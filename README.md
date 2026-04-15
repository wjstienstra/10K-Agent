# AI-Driven Strategic Equity Analyst (RAG)

A specialized Retrieval-Augmented Generation (RAG) application designed to transform raw SEC 10-K filings into high-level investment memos. This tool bridges the gap between unstructured financial regulatory filings and actionable investment insights.

## 🚀 The Core Challenge
SEC filings (10-Ks) are massive HTML/ASCII containers that often include noisy data like binary artifacts and complex nesting. Standard RAG pipelines often struggle with this "noise," leading to diluted AI context and halluncinations.

**My Solution:** I engineered a custom preprocessing pipeline using **BeautifulSoup** and **Regex** to strip out non-textual artifacts and isolate critical financial sections, resulting in significantly improved context relevance and analysis speed.

## 🛠️ Technical Stack
* **LLM Orchestration:** LangChain
* **Models:** Google Gemini 1.5 Flash & Pro (Optimized for financial reasoning)
* **Vector Database:** ChromaDB (Utilizing collection-based isolation per ticker)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local execution for performance and privacy)
* **Frontend:** Streamlit
* **Data Sourcing:** SEC EDGAR API

## ✨ Key Features
* **Financial Signal Extraction:** Automatically identifies the relationship between Net Income, Operating Income, and Free Cash Flow (FCF).
* **Automated 'Moat' & Risk Analysis:** Uses semantic search to evaluate competitive advantages and geopolitical/regulatory risks.
* **Data Isolation:** Strict metadata filtering ensures that data from different companies or years does not get mixed during analysis.
* **Developer Utilities:** Includes custom inspection tools (in the `/scripts` folder) to validate vector store integrity and monitor API model availability.

## 📁 Project Structure
* `app.py`: Streamlit frontend and UI logic.
* `main.py`: Core RAG logic and AI agent synthesis.
* `ingestion.py`: Data cleaning (BeautifulSoup) and vector database management.
* `tools.py`: Automated SEC document downloader.
* `/scripts`: Maintenance tools for database inspection and API testing.

## ⚙️ Installation & Usage
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Add your `GOOGLE_API_KEY` to a `.env` file.
4.  Run the application:
    ```bash
    streamlit run app.py
    ```