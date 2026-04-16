import time
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# De Verbeterde Nederlandse Hedge Fund Analyst Prompt
MASTER_INVESTMENT_PROMPT = """
ROL: Je bent een Senior Equity Research Analyst, gespecialiseerd in SEC 10-K analyses.
TAAK: Schrijf een kritisch en feitelijk investeringsmemo op basis van de verstrekte documentfragmenten.

STRIKTE RICHTLIJNEN:
1. TAAL: Reageer uitsluitend in het Nederlands.
2. BRONVERMELDING: Gebruik alleen de verstrekte context. Citeer jaartallen (bijv. "In 2023...").
3. INTEGRITEIT: Als specifieke data (zoals 'Net Debt' of een margenummer) niet in de fragmenten staat, schrijf dan "Niet aangetroffen in de verstrekte context". Ga NIET schatten of hallucineren.
4. TOON: Zakelijk, objectief en kritisch.
5. DATA-DETECTIE: Zoek specifiek naar getallenreeksen die volgen na woorden als 'Revenue', 'Operating Income' of 'Net Income'. De getallen staan vaak verderop in de tekst door de conversie.

STRUCTUUR:
1. **Financiële Snapshot (Tabel)**: Maak een tabel met de kolommen [Jaar, Omzet, Operationeel Inkomen, Brutomarge %, Vrije Kasstroom].
2. **Economische Moat**: Evalueer het concurrentievoordeel (Netwerkeffecten, Wisselkosten, Merksterkte, of Kostenvoordeel).
3. **Risico-Audit**: Benoem de 3 belangrijkste risico's die expliciet in de documenten worden genoemd.
4. **Conclusie**: Classificatie (Groei / Volwassen / Risicovol) + een korte onderbouwing van één zin.

LET OP: Financiële tabellen kunnen door de extractie verminkt zijn (cijfers achter elkaar in plaats van in kolommen). Zoek naar patronen zoals 'Revenue ... [cijfer1] [cijfer2] [cijfer3]' waarbij de cijfers van rechts naar links vaak 2024, 2023, 2022 vertegenwoordigen.

"""

def clean_document_content(doc):
    content = doc.page_content
    
    # 1. Harde stop bij technische metadata
    noise_markers = ["extras':", "signature':", "'signature':", "document_id", "file_name"]
    for marker in noise_markers:
        if marker in content:
            content = content.split(marker)[0]
    
    # 2. Verwijder HTML-achtige tags
    content = re.sub(r'<[^>]*>', '', content)

    content = re.sub(r'[a-z]+:[a-z]+="[^"]*"', '', content) 
    content = re.sub(r'\{[^\}]*\}', '', content) # Verwijder JSON-achtige restanten
    
    # 3. Verwijder extreem lange strings (base64/signatures)
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

    # Slimme zoekopdrachten
    # We zoeken specifiek op de exacte kop van Item 8 (de financiele sectie)
    numeric_docs = vector_db.similarity_search(
        "ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA CONSOLIDATED STATEMENTS OF OPERATIONS", 
        k=15 
    )

    # Zoekopdracht 2: Focus op de business kwaliteit (Item 1 & 1A in 10-K)
    strategy_docs = vector_db.similarity_search(
        "Business Overview, Item 1, Competitive Moat, Trademarks, Competition and Risk Factors", 
        k=5
    )
    
    all_docs = numeric_docs + strategy_docs
    all_docs.sort(key=lambda x: x.metadata.get('year', '0000'))
    
    context_list = []
    for d in all_docs:
        clean_text = clean_document_content(d)
        year = d.metadata.get('year', 'Onbekend')
        context_list.append(f"--- DOCUMENT FRAGMENT JAAR {year} ---\n{clean_text}")
    
    context = "\n\n".join(context_list)
    
    enhanced_prompt = f"""
    CONTEXT UIT 10-K DOCUMENTEN:
    {context}
    
    OPDRACHT:
    {MASTER_INVESTMENT_PROMPT}
    """

    MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest"]
    
    for model_name in MODELS:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name, 
                temperature=0.0, 
                timeout=120, 
                max_retries=3
            ) 
            response = llm.invoke(enhanced_prompt)
            
            # Fix voor 'list object' error: dwing content naar string
            if hasattr(response, 'content'):
                raw_content = response.content
                if isinstance(raw_content, list):
                    final_answer = ""
                    for item in raw_content:
                        if isinstance(item, dict) and 'text' in item:
                            final_answer += item['text']
                    if not final_answer:
                        final_answer = str(raw_content)
                else:
                    final_answer = str(raw_content)
            else:
                final_answer = str(response)

            return final_answer.replace("$", "\\$")
            
        except Exception as e:
            if model_name == MODELS[-1]:
                raise e
            time.sleep(2)
            continue

def ask_ai_question(company_id, user_question):
    """Beantwoordt vragen en handelt de lijst-respons van de API correct af."""
    ticker = company_id.replace("_report", "").upper()
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./db", embedding_function=embeddings, collection_name=company_id)
    
    docs = vector_db.similarity_search(user_question, k=5)
    context_list = []
    for d in docs:
        clean_text = clean_document_content(d)
        context_list.append(f"[Bron: {d.metadata.get('year', 'Onbekend')}] {clean_text}")
    
    context = "\n\n".join(context_list)
    
    chat_prompt = f"""
    Je bent een feitelijke SEC-assistent. Je taak is om de vraag van de gebruiker te beantwoorden 
    op basis van de onderstaande 10-K fragmenten van {ticker}.

    REGELS:
    1. ANTWOORD ALTIJD IN HET NEDERLANDS.
    2. Als het antwoord niet in de verstrekte tekst staat, zeg dan: "Ik kan het antwoord op deze vraag niet vinden in de geanalyseerde jaarverslagen van {ticker}."
    3. Citeer altijd het jaar van de bron: (Bron: 2023).
    4. Gebruik geen externe kennis over het bedrijf die niet in de fragmenten staat.

    CONTEXT:
    {context}

    VRAAG VAN GEBRUIKER:
    {user_question}
    """
    
    llm = ChatGoogleGenerativeAI(model="gemini-flash-lite-latest", temperature=0.3, timeout=60)
    response = llm.invoke(chat_prompt)
    
    # --- FIX: TEXT EXTRACTIE UIT LIJST ---
    raw_content = response.content if hasattr(response, 'content') else response
    
    if isinstance(raw_content, list):
        final_text = ""
        for item in raw_content:
            if isinstance(item, dict) and 'text' in item:
                final_text += item['text']
        content = final_text if final_text else str(raw_content)
    else:
        content = str(raw_content)
        
    return content.replace("$", "\\$")

get_analysis_section = lambda cid, sec: get_comprehensive_analysis(cid)