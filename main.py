import os
import time
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# De Ultimate Hedge Fund Analyst Prompt
MASTER_INVESTMENT_PROMPT = """
Je bent een Senior Hedge Fund Analyst. Je krijgt fragmenten uit 10-K jaarverslagen van de afgelopen 5 jaar. 
Schrijf een kritisch investeringsmemo.

STRUCTUUR:
1. **Financiële Snapshot (Tabel):** Maak een tabel met Revenue, Gross Margin (%), Operating Income en Net Debt voor alle beschikbare jaren.
2. **De 'Moat' & Business Model:** Hoe beschermt het bedrijf zijn winst? Wordt de competitieve positie sterker of zwakker?
3. **Kapitaalallocatie:** Waar gaat het geld heen? (R&D, Buybacks, Overnames). Is dit in het belang van de aandeelhouder?
4. **Risico-Audit:** Zoek naar veranderende risico's in de laatste 2 jaar (AI, regelgeving, concurrentie).
5. **Conclusie:** Kwaliteit van de winst en fase-classificatie [Growth / Mature / Cyclical / Turnaround].

RICHTLIJNEN:
- Citeer jaartallen (bijv. "In 2022...").
- Wees kritisch en zakelijk.
- Gebruik Markdown voor een strakke opmaak.
"""

def clean_document_content(doc):
    content = doc.page_content
    
    # 1. Harde stop bij technische metadata
    noise_markers = ["extras':", "signature':", "'signature':", "document_id", "file_name"]
    for marker in noise_markers:
        if marker in content:
            content = content.split(marker)[0]
    
    # 2. Verwijder HTML-achtige tags die vaak in SEC-filings blijven hangen
    content = re.sub(r'<[^>]*>', '', content)
    
    # 3. Verwijder extreem lange strings zonder spaties (zoals base64 signatures)
    words = content.split()
    clean_words = [w for w in words if len(w) < 50]
    content = " ".join(clean_words)
    
    return content.strip()

def get_comprehensive_analysis(company_id):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./db", embedding_function=embeddings, collection_name=company_id)
    
    count = vector_db._collection.count()
    if count == 0:
        return "❌ Fout: De database is leeg of de collectie-naam matcht niet."

    print(f"🕵️ Agent start onderzoek in {count} fragmenten...")

    # VERBETERING: Meer fragmenten (k=10) nu de data schoon is.
    # We zoeken nu specifiek naar de 'Selected Financial Data' tabelkoppen.
    numeric_docs = vector_db.similarity_search(
        "Financial Highlights Table Selected Financial Data Revenue Net Income Operating Income Cash Flow Balance Sheet", 
        k=6
    )
    
    strategy_docs = vector_db.similarity_search(
        "Business strategy item 1A risk factors competition moat product development AI innovation", 
        k=6
    )
    
    # Combineer en sorteer de documenten op jaar uit de metadata
    all_docs = numeric_docs + strategy_docs
    all_docs.sort(key=lambda x: x.metadata.get('year', '0000'))
    
    # --- DATA SANITIZATION LAYER ---
    context_list = []
    for d in all_docs:
        clean_text = clean_document_content(d)
        year = d.metadata.get('year', 'Onbekend')
        # We voegen een duidelijke kop toe per fragment
        context_list.append(f"--- DOCUMENT FRAGMENT JAAR {year} ---\n{clean_text}")
    
    context = "\n\n".join(context_list)
    
    # We voegen een extra instructie toe aan de prompt om 'N/A' te vermijden
    enhanced_prompt = f"""
    CONTEXT UIT 10-K DOCUMENTEN:
    {context}
    
    BELANGRIJKE INSTRUCTIE: 
    Kijk goed naar de fragmenten gemarkeerd met 'DOCUMENT FRAGMENT JAAR'. 
    Probeer de numerieke waarden uit de tekst en tabellen te extraheren voor de gevraagde tabel.
    
    OPDRACHT:
    {MASTER_INVESTMENT_PROMPT}
    """

    MODELS = ["gemini-pro-latest", "gemini-flash-latest", "gemini-flash-lite-latest"]
    
    for model_name in MODELS:
        try:
            # We verlagen de temperature nog iets meer voor maximale feitelijke nauwkeurigheid
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0) 
            response = llm.invoke(enhanced_prompt)
            
            final_answer = response.content 
            return final_answer.replace("$", "\\$")
            
        except Exception as e:
            if "429" in str(e):
                continue
            if model_name == MODELS[-1]:
                raise e

def ask_ai_question(company_id, user_question):
    """Beantwoordt een specifieke vraag over de opgeslagen 10-K's met cleanup."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./db", embedding_function=embeddings, collection_name=company_id)
    
    # We zoeken nu in de 'schone' database
    docs = vector_db.similarity_search(user_question, k=5)
    
    context_list = []
    for d in docs:
        clean_text = clean_document_content(d)
        context_list.append(f"[Bron: {d.metadata.get('year', 'Onbekend')}] {clean_text}")
    
    context = "\n\n".join(context_list)
    
    chat_prompt = f"""
    Je bent een behulpzame AI-analist. Gebruik de onderstaande fragmenten uit 10-K jaarverslagen. 
    Citeer altijd het jaartal van je bron.

    CONTEXT:
    {context}

    VRAAG:
    {user_question}
    """
    
    # Gebruik Flash voor snelheid in de chat
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.3)
    response = llm.invoke(chat_prompt)
    
    # CRUCIAL: Pak alleen de tekstuele content
    if hasattr(response, 'content'):
        content = response.content
    else:
        content = str(response)
        
    return content.replace("$", "\\$")

# Voor compatibiliteit met app.py
get_analysis_section = lambda cid, sec: get_comprehensive_analysis(cid)