import os
import re
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ESSENTIEEL: Laad de omgevingsvariabelen direct aan het begin
load_dotenv()

def clean_html_content(raw_html):
    """
    Gebruikt BeautifulSoup om HTML-tags te verwijderen. 
    We gebruiken 'lxml' of 'html.parser' voor snelheid.
    """
    # Sommige SEC bestanden zijn enorm, we breken ze op als ze geen HTML lijken
    if "<HTML" not in raw_html.upper() and "<DOCUMENT" not in raw_html.upper():
        return raw_html[:50000] # Fallback voor platte tekst

    soup = BeautifulSoup(raw_html, "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    
    # Gebruik een newline separator om tabel-data niet aan elkaar te plakken
    return soup.get_text(separator='\n')

def run_multi_year_ingestion(directory_path, ticker, status_element=None):
    all_documents = []
    ticker = ticker.upper()
    
    # Gebruik een regex om alleen de tekst/html secties uit de container te vissen
    # Dit negeert alle binaire PDF/IMAGE rommel onderaan het bestand
    doc_re = re.compile(r'<DOCUMENT>(.*?)</DOCUMENT>', re.DOTALL | re.IGNORECASE)
    type_re = re.compile(r'<TYPE>(.*?)\n', re.IGNORECASE)

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower() == "full-submission.txt":
                path = os.path.normpath(os.path.join(root, file))
                
                # Veilige jaar extractie
                folder_name = os.path.basename(root)
                year = "Onbekend"
                parts = folder_name.split('-')
                if len(parts) > 1 and parts[1].isdigit():
                    y_val = int(parts[1])
                    year = f"20{parts[1]}" if y_val < 50 else f"19{parts[1]}"

                print(f"📄 Extractie start: {ticker} {year}...")

                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    # Vind alle document-secties
                    documents = doc_re.findall(content)
                    clean_text_parts = []

                    for doc in documents:
                        # Check het type van het document (we willen alleen 10-K of HTML)
                        doc_type = type_re.search(doc)
                        if doc_type:
                            t = doc_type.group(1).strip().upper()
                            # We skippen types als GRAPHIC, PDF, ZIP etc.
                            if any(x in t for x in ["10-K", "HTML", "TEXT"]):
                                # Nu pas BeautifulSoup op DIT specifieke fragment
                                soup = BeautifulSoup(doc, "html.parser")
                                # Haal de tekst op
                                clean_text_parts.append(soup.get_text(separator=' '))

                    final_text = "\n\n".join(clean_text_parts)
                    # Basis filtering voor rommel-regels
                    final_lines = [l.strip() for l in final_text.splitlines() if len(l.strip()) > 10]
                    final_content = "\n".join(final_lines)

                    if len(final_content) > 500:
                        all_documents.append(Document(
                            page_content=final_content[:100000], # Beperk de analyse tot de eerste 100k tekens, waar doorgaans de belangrijke informatie staat omwillen van de snelheid.
                            metadata={"year": str(year), "ticker": ticker}
                        ))
                        print(f"✅ Geladen: {year} ({len(final_content)} tekens)")

                except Exception as e:
                    print(f"❌ Fout bij {year}: {e}")

    print(f"\n📊 Totaal aantal verslagen geladen: {len(all_documents)}")

    if not all_documents:
        print("❌ FOUT: Geen bruikbare documenten gevonden. Controleer je brondata.")
        return

    # Splitting & Database
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_documents(all_documents)
    
    print(f"🚀 Database vullen voor {ticker} met {len(chunks)} fragmenten...")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    collection_name = f"{ticker}_report"
    
    vector_db = Chroma(
        persist_directory="./db", 
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    # We voegen de documenten toe
    vector_db.add_documents(chunks)
    
    if status_element:
        status_element.success(f"🎉 {ticker} succesvol geïndexeerd!")