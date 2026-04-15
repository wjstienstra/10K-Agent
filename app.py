import streamlit as st
import os
import time
from ingestion import run_multi_year_ingestion
from tools import fetch_multi_year_10k, get_all_sec_tickers
from main import get_comprehensive_analysis, ask_ai_question

# --- CONFIGURATIE ---
st.set_page_config(page_title="Equity Research AI", layout="wide", page_icon="📈")

# Helper om jaren te tonen
def get_indexed_years(ticker):
    """Checkt welke jaren er daadwerkelijk in de metadata van de DB staan."""
    filings_path = f"sec-edgar-filings/{ticker.upper()}/10-K"
    if os.path.exists(filings_path):
        folders = os.listdir(filings_path)
        years = []
        for f in folders:
            if "-" in f:
                parts = f.split("-")
                if len(parts) > 1:
                    year_part = parts[1]
                    years.append(f"20{year_part}" if int(year_part) < 50 else f"19{year_part}")
        return sorted(list(set(years)), reverse=True)
    return []

def check_if_indexed(ticker):
    """Checkt of de collectie bestaat EN of er daadwerkelijk chunks in zitten."""
    db_path = "./db"
    if not os.path.exists(db_path):
        return False
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import Chroma
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        collection_name = f"{ticker.upper()}_report"
        
        vector_db = Chroma(
            persist_directory=db_path, 
            embedding_function=embeddings, 
            collection_name=collection_name
        )
        
        return vector_db._collection.count() > 0
    except Exception:
        return False

# --- UI ELEMENTEN ---
st.title("🤖 Strategic Equity Analyst")
st.caption("Gevalideerde 10-K Analyse & Data Integriteit Check")

all_tickers = get_all_sec_tickers()
ticker = None 

if 'auto_analyze' not in st.session_state:
    st.session_state.auto_analyze = False

with st.sidebar:
    st.header("🏢 Bedrijf Selecteren")
    
    selected_option = st.selectbox(
        "Zoek een ticker:",
        options=all_tickers,
        index=None,
        placeholder="Typ bijv. 'NVIDIA'..."
    )
    
    if selected_option:
        ticker = selected_option.split(" - ")[0].upper()
        
        st.subheader("⚙️ Analyse Instellingen")
        lookback_years = st.slider(
            "Lookback periode (jaren):", 
            min_value=1, 
            max_value=5, 
            value=3,
            help="Meer jaren zorgen voor betere trendanalyse."
        )
        
        is_indexed = check_if_indexed(ticker)
        indexed_years = get_indexed_years(ticker)
        
        if not is_indexed:
            st.info(f"💡 {ticker} moet nog geïndexeerd worden.")
            if st.button("📥 Download & Indexeer", type="primary"):
                with st.status(f"Project {ticker} initialiseren...", expanded=True) as s:
                    s.write(f"🌐 Stap 1: Data ophalen...")
                    data_path = fetch_multi_year_10k(ticker, amount=lookback_years)
                    
                    s.write("✂️ Stap 2: Indexeren (CPU intensief)...")
                    run_multi_year_ingestion(data_path, ticker, status_element=s)
                    
                    s.update(label=f"✅ {ticker} is gereed!", state="complete")
                
                st.session_state.auto_analyze = True
                st.rerun()
        else:
            st.success(f"✅ Data aanwezig voor: {', '.join(indexed_years)}")
            if st.button("🚀 Start Analyse"):
                st.session_state.auto_analyze = True

# --- HOOFDSCHERM: AUTOMATISCHE ANALYSE ---
if ticker and (st.session_state.auto_analyze or check_if_indexed(ticker)):
    st.header(f"Strategisch Rapport: {selected_option}")
    
    try:
        with st.spinner(f"Agent voert deep-dive uit op {ticker}..."):
            company_id = f"{ticker.upper()}_report"
            report_text = get_comprehensive_analysis(company_id)

            # EXTRA VEILIGHEID: Dwing output naar tekst als het een dictionary/lijst is
            if isinstance(report_text, list):
                report_text = report_text[0]
            if isinstance(report_text, dict):
                report_text = report_text.get('text', report_text.get('content', str(report_text)))
            
            with st.container():
                st.markdown(f"### 📈 Fundamentele Analyse: {ticker}")
                available_years = get_indexed_years(ticker)
                st.caption(f"Geanalyseerde periodes: {', '.join(available_years)}")
                
                st.info("De AI heeft specifiek gezocht naar 'Consolidated Statements' en 'Item 1A' risico's.")
                st.markdown(report_text)
            
            st.divider()
            st.download_button(
                label="💾 Download Volledig Rapport",
                data=report_text,
                file_name=f"Analyse_{ticker}_MultiYear.txt",
                mime="text/plain"
            )
            st.session_state.auto_analyze = False
            
    except Exception as e:
        st.error(f"Fout tijdens de synthese: {e}")
        st.session_state.auto_analyze = False

# --- CHATBOT SECTIE ---
if ticker and check_if_indexed(ticker):
    st.divider()
    st.subheader(f"💬 Stel een vraag over {ticker}")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Stel een vraag over {ticker}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("AI onderzoekt documenten..."):
                try:
                    company_id = f"{ticker}_report"
                    answer = ask_ai_question(company_id, prompt)
                    
                    # Ook hier opschonen van eventuele object-output
                    if isinstance(answer, dict):
                        answer = answer.get('text', str(answer))
                        
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Chatfout: {e}")