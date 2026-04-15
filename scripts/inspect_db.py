# Script om een check te doen om de embedding goed gaat en er daadwerkelijk iets in de db terechtkomt. Gebruikt tijdens het bouwen.

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Laad variabelen (nodig als je paden uit .env haalt)
load_dotenv()

def inspect_database(ticker):
    """
    Inspecteert de specifieke collectie van een bedrijf in de database.
    """
    # 1. Gebruik EXACT dezelfde embeddings als in je ingestion.py
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 2. Specificeer de juiste collectie (bijv. NVDA_report)
    collection_name = f"{ticker.upper()}_report"
    
    if not os.path.exists("./db"):
        print("❌ De map './db' bestaat niet. Run eerst de ingestion.")
        return

    vector_db = Chroma(
        persist_directory="./db", 
        embedding_function=embeddings,
        collection_name=collection_name
    )

    # 3. Check het aantal items
    try:
        collection_count = vector_db._collection.count()
        print(f"\n--- Database Inspectie: {collection_name} ---")
        print(f"Aantal fragmenten in deze collectie: {collection_count}")

        if collection_count > 0:
            # Bekijk de eerste 2 fragmenten
            sample = vector_db._collection.get(limit=2)
            print("\nVoorbeeld van de opgeslagen tekst en metadata:")
            print("-" * 50)
            for i in range(len(sample['documents'])):
                text = sample['documents'][i]
                metadata = sample['metadatas'][i]
                print(f"📄 Bron Jaar: {metadata.get('year', 'Onbekend')}")
                print(f"📝 Tekst: {text[:250]}...")
                print("-" * 50)
        else:
            print(f"⚠️ De collectie '{collection_name}' is leeg.")
            
    except Exception as e:
        print(f"❌ Fout bij het benaderen van de collectie: {e}")

if __name__ == "__main__":
    # Test het script voor bijvoorbeeld Nvidia
    ticker_to_check = input("Welke ticker wil je inspecteren? (bijv. NVDA): ")
    inspect_database(ticker_to_check)