import os
import json
from dotenv import load_dotenv
from chromadb.config import Settings
from langchain_gigachat.embeddings.gigachat import GigaChatEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv(override=True)

# --- Global Variables ---
GIGA_CREDENTIALS = os.environ.get("GIGACHAT_TOKEN")
PERSIST_DIRECTORY_QUERIES = "chroma_db_successful_queries" 

embeddings_giga = None
db_queries = None
# Using a distinct collection name for these query-specific RAG entries
RAG_QUERY_COLLECTION_NAME = "successful_text2sql_queries_v1" 

# --- Initialization ---
def init_rag_db():
    """
    Initializes the GigaChat embeddings and Chroma DB for storing successful queries.
    Returns True on success, False on failure.
    """
    global embeddings_giga, db_queries

    if not GIGA_CREDENTIALS:
        print("[RAG HANDLER ERROR] GIGACHAT_TOKEN not found in environment variables.")
        return False

    try:
        embeddings_giga = GigaChatEmbeddings(
            credentials=GIGA_CREDENTIALS, verify_ssl_certs=False
        )
    except Exception as e:
        print(f"[RAG HANDLER ERROR] Failed to initialize GigaChatEmbeddings: {e}")
        return False

    client_settings = Settings(anonymized_telemetry=False)
    
    # Ensure the persist directory exists
    os.makedirs(PERSIST_DIRECTORY_QUERIES, exist_ok=True)

    try:
        db_queries = Chroma(
            collection_name=RAG_QUERY_COLLECTION_NAME,
            persist_directory=PERSIST_DIRECTORY_QUERIES,
            embedding_function=embeddings_giga,
            client_settings=client_settings
        )
        # Attempt a count to confirm collection is accessible and "warm it up".
        count = db_queries._collection.count()
        print(f"[RAG HANDLER] Chroma DB initialized. Collection '{RAG_QUERY_COLLECTION_NAME}' count: {count}. Path: {PERSIST_DIRECTORY_QUERIES}")
        return True
    except Exception as e:
        print(f"[RAG HANDLER ERROR] Failed to initialize or load Chroma DB collection '{RAG_QUERY_COLLECTION_NAME}' from {PERSIST_DIRECTORY_QUERIES}: {e}")
        db_queries = None
        return False

# --- Core Functions ---
def add_to_rag_db(user_prompt: str, formal_prompt: str, plan: str, sql_query: str, final_answer: str):
    """
    Adds a successful query interaction to the RAG database.
    Returns True on success, False on failure.
    """
    global db_queries
    print(f"[RAG HANDLER DEBUG] add_to_rag_db called with user_prompt: {user_prompt[:50]}...")
    if db_queries is None:
        print("[RAG HANDLER ERROR] RAG DB not initialized. Call init_rag_db() first. (inside add_to_rag_db)")
        return False

    metadata = {
        "user_prompt": user_prompt,
        "formal_prompt": formal_prompt,
        "plan": plan,
        "sql_query": sql_query,
        "final_answer": final_answer,
    }
    
    try:
        print(f"[RAG HANDLER DEBUG] Attempting to add text to ChromaDB. Metadata: {json.dumps(metadata, ensure_ascii=False)[:100]}...")
        # Chroma with persist_directory handles persistence automatically on writes.
        db_queries.add_texts(texts=[user_prompt], metadatas=[metadata])
        print(f"[RAG HANDLER DEBUG] Successfully called db_queries.add_texts. User prompt: {user_prompt[:50]}...")
        # print(f"[RAG HANDLER] New collection count: {db_queries._collection.count()}") # Optional: log count after add
        return True
    except Exception as e:
        print(f"[RAG HANDLER ERROR] Failed to add to RAG DB during add_texts operation: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for more context
        return False


def query_rag_db(user_query: str, n_results: int = 1,
                 score_threshold_exact: float = 0.05, 
                 score_threshold_similar: float = 0.25):
    """
    Queries the RAG database for similar past interactions.
    Scores are distances (lower is more similar).
    """
    global db_queries
    if db_queries is None:
        print("[RAG HANDLER ERROR] RAG DB not initialized. Call init_rag_db() first.")
        return {"status": "error", "message": "RAG DB not initialized."}

    if not user_query or not user_query.strip():
        print("[RAG HANDLER] Empty user query received.")
        return {"status": "no_match", "data": {}}

    try:
        # print(f"[RAG HANDLER] Querying RAG DB for: {user_query[:50]}...")
        results_with_scores = db_queries.similarity_search_with_score(
            query=user_query,
            k=n_results 
        )
        # print(f"[RAG HANDLER] Raw results from Chroma: {results_with_scores}")

        if not results_with_scores:
            # print("[RAG HANDLER] No similar documents found in RAG DB.")
            return {"status": "no_match", "data": {}}

        best_match_doc, best_match_score = results_with_scores[0]
        
        # print(f"[RAG HANDLER] Best match -> Score: {best_match_score:.4f}, Doc Prompt: {best_match_doc.metadata.get('user_prompt', '')[:50]}...")

        if best_match_score <= score_threshold_exact:
            # print(f"[RAG HANDLER] Exact match found (score: {best_match_score:.4f})")
            return {"status": "exact", "data": best_match_doc.metadata}
        elif best_match_score <= score_threshold_similar:
            # print(f"[RAG HANDLER] Similar match found (score: {best_match_score:.4f})")
            return {"status": "similar", "data": best_match_doc.metadata}
        else:
            # print(f"[RAG HANDLER] Match found but score {best_match_score:.4f} (>{score_threshold_similar}) is not similar enough.")
            return {"status": "no_match", "data": {}}

    except Exception as e:
        print(f"[RAG HANDLER ERROR] Failed to query RAG DB: {e}")
        return {"status": "error", "message": f"Error querying RAG DB: {e}"}

# --- Example Usage (for testing this module directly) ---
if __name__ == '__main__':
    print("Running RAG Handler example...")
    
    # Attempt to initialize the DB
    if not init_rag_db():
        print("Exiting example due to RAG DB initialization failure.")
        exit(1) # Exit with an error code
        
    # At this point, db_queries should be initialized if init_rag_db() returned True
    print(f"Initial DB count for collection '{RAG_QUERY_COLLECTION_NAME}': {db_queries._collection.count()}")

    # Test add
    sample_data = [
        {
            "user_prompt": "What is the average salary in Maykop for 2023?",
            "formal_prompt": "Calculate the average salary in Maykop for the year 2023.",
            "plan": "1. Filter salary data for Maykop. 2. Filter for year 2023. 3. Calculate average of s.value.",
            "sql_query": "SELECT AVG(s.value) FROM salary s WHERE UPPER(s.municipal_district_name) LIKE '%МАЙКОП%' AND s.year = 2023",
            "final_answer": "The average salary in Maykop for 2023 was 50000 RUB."
        },
        {
            "user_prompt": "Population of Adygeysk in 2024?",
            "formal_prompt": "Find the total population of Adygeysk for the year 2024.",
            "plan": "1. Filter population data for Adygeysk. 2. Filter for year 2024. 3. Sum p.value.",
            "sql_query": "SELECT SUM(p.value) FROM population p WHERE UPPER(p.municipal_district_name) LIKE '%АДЫГЕЙСК%' AND p.year = 2024 AND p.age = 'Все возраста'",
            "final_answer": "The total population of Adygeysk in 2024 is 15000."
        }
    ]

    num_added = 0
    for i, item in enumerate(sample_data):
        if add_to_rag_db(**item):
            print(f"Added sample item {i+1} successfully.")
            num_added +=1
        else:
            print(f"Failed to add sample item {i+1}.")
    
    if num_added > 0:
         print(f"DB count after adds: {db_queries._collection.count()}")

    # Test queries
    test_queries = [
        "What is the average salary in Maykop for 2023?", # Should be exact
        "Average salary in Maykop city, 2023",             # Should be similar
        "avg salary Maykop 2023",                          # Potentially similar
        "Population of Adygeysk in 2024?",                 # Should be exact
        "Adygeysk population 2024",                        # Should be similar
        "Show me migration trends for young people in Lomonosovskiy district.", # Should be no_match
        "  ",                                              # Empty query
        "What is love?"                                    # No match
    ]

    print("\n--- Query Tests ---")
    for q_idx, user_q in enumerate(test_queries):
        print(f"\nTest Query {q_idx+1}: \"{user_q}\"")
        result = query_rag_db(user_q)
        # Pretty print JSON for readability
        print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n--- RAG Handler Example Done ---")
