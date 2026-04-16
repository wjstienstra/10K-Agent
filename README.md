# AI-Driven Strategic Equity Analyst (RAG)

A specialized Retrieval-Augmented Generation (RAG) application designed to transform raw SEC 10-K filings into high-level investment memos. This tool bridges the gap between unstructured financial regulatory filings and actionable investment insights using the Gemini 1.5 ecosystem.

## 🚀 The Core Challenge
SEC filings (10-Ks) are massive HTML/ASCII containers that often include noisy data like binary artifacts and complex nesting. Standard RAG pipelines often struggle with this "noise," leading to diluted AI context and hallucinations.

**My Solution:** I engineered a custom preprocessing pipeline using **BeautifulSoup** and **Regex** to strip out non-textual artifacts and isolate critical financial sections, resulting in significantly improved context relevance for qualitative analysis.

## 🔍 Technical Analysis: The RAG "Table-Trap"
During development, this project highlighted a fundamental frontier in RAG-architectures: **Structure Loss in Unstructured Parsing**.

* **The Issue:** Standard PDF-to-text parsers flatten table structures. Once a financial table (like the *Consolidated Statement of Operations*) is converted to raw text, the spatial relationship between labels ("Revenue") and values ("$394,332") is often corrupted, making multi-year numerical extraction unreliable.
* **The Success:** The agent excels at **Qualitative Synthesis**. It successfully identifies complex strategic risks (e.g., Apple's Google-search licensing threats) and competitive moats that are buried in narrative text.

## 🗺️ Roadmap: Towards a Hybrid v2.0
To move from a proof-of-concept to a production-grade financial tool, the next iteration will implement a **Hybrid Agentic Architecture**:

1.  **Structured Data Integration:** Using financial APIs (e.g., Financial Modeling Prep or Alpha Vantage) to fetch 100% accurate numerical tables.
2.  **Advanced Layout Parsing:** Replacing standard parsers with **Docling (IBM)** or **Unstructured.io** to convert PDFs into **Markdown**. This preserves table hierarchies for the LLM.
3.  **Agentic Tool Use:** Transitioning from simple RAG to a **ReAct Agent** that can choose between "Search 10-K" for context and "Query API" for hard numbers.

## 🛠️ Technical Stack
* **LLM Orchestration:** LangChain
* **Models:** Google Gemini Flash & Pro (Optimized for financial reasoning)
* **Vector Database:** ChromaDB (Collection-based isolation per ticker)
* **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local execution)
* **Frontend:** Streamlit
* **Data Sourcing:** SEC EDGAR API

## ✨ Key Features
* **Strategic Signal Extraction:** Automatically identifies competitive advantages and geopolitical risks.
* **Multi-Year Context:** Analyzes trends across multiple filings to identify shifting management sentiment.
* **Data Integrity Guardrails:** Implemented strict "No-Hallucination" instructions; the agent reports "Data Not Found" rather than inventing financial figures.

## 📁 Project Structure
* `app.py`: Streamlit frontend with dynamic status updates.
* `main.py`: Core RAG logic and "Analyst" prompt engineering.
* `ingestion.py`: Layout-aware cleaning and vector management.
* `/scripts`: Maintenance tools for DB inspection and API monitoring.

---
*Developed as a technical demonstration of RAG capabilities, Prompt Engineering, and Financial Data ETL pipelines.*