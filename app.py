# app.py (Updated for MySQL, Keyword-Enhanced Semantic Search, and Advanced Prompting)

import os
import json
import logging
import mysql.connector
from mysql.connector import Error

from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Core Dependencies for Semantic Search ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Core Dependencies for LLM ---
from groq import Groq

# --- SETUP AND CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

## SECURITY WARNING: Hardcoding credentials is not recommended for production.
## Use environment variables for API keys and database passwords.
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = 'llama3-70b-8192'

# --- DATABASE CONFIGURATION ---
db_config = {
    'host': '195.35.59.66',
    'database': 'u914092009_chat_bot',
    'user': 'u914092009_vidhan',
    'password': '#Vcr@2001'
}

# --- SEMANTIC SEARCH SETUP ---
# Load the model only once when the app starts
logger.info("Loading sentence transformer model for semantic search...")
SEARCH_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
logger.info("Sentence transformer model loaded.")

# --- IN-MEMORY CACHE FOR FAQS FROM DB ---
# This avoids hitting the DB for all FAQs on every user query
CACHED_FAQS = []
FAQ_EMBEDDINGS = None

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)
CORS(app)

# ==============================================================================
# == DATABASE & KNOWLEDGE BASE LOADING
# ==============================================================================

def init_db():
    """Checks the MySQL database connection and ensures necessary tables exist."""
    conn = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Ensure chat_history table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                context TEXT,
                keywords TEXT,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_query (query(255))
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        ''')

        # Check if the faqs table exists and has the keywords column
        cursor.execute("SHOW COLUMNS FROM faqs LIKE 'keywords';")
        result = cursor.fetchone()
        if not result:
            logger.warning("The 'keywords' column does not exist in the 'faqs' table. Semantic search may be less effective.")
            logger.info("Consider running: ALTER TABLE faqs ADD COLUMN keywords TEXT NULL;")

        conn.commit()
        logger.info(f"Database connection to '{db_config['database']}' successful and tables are ready.")
    except Error as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        raise
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

def load_faqs_from_db():
    """
    Loads FAQs from the MySQL DB into memory and pre-computes embeddings
    using both the question and the keywords for improved search accuracy.
    """
    global CACHED_FAQS, FAQ_EMBEDDINGS
    conn = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # Fetch all necessary fields, including the new keywords column
        cursor.execute("SELECT faq_id, question, answer, keywords FROM faqs")
        CACHED_FAQS = cursor.fetchall()
        
        if not CACHED_FAQS:
            logger.warning("No FAQs found in the database. The chatbot will have no knowledge base.")
            return

        # *** KEY IMPROVEMENT ***
        # Create a combined text for embedding. This makes the search more powerful.
        # For example: "What is the dead volume? dead volume, bulk bottles, reagent volume"
        texts_to_embed = [
            f"{faq['question']} {faq.get('keywords', '')}".strip() for faq in CACHED_FAQS
        ]
        
        FAQ_EMBEDDINGS = SEARCH_MODEL.encode(texts_to_embed, show_progress_bar=True)
        logger.info(f"Successfully loaded and created embeddings for {len(CACHED_FAQS)} FAQs from the database.")

    except Error as e:
        logger.error(f"Could not load FAQs from the database: {e}", exc_info=True)
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()

# ==============================================================================
# == SEMANTIC SEARCH FUNCTION
# ==============================================================================

def get_relevant_context(query: str, top_k=3) -> str:
    """Finds the most relevant FAQs using keyword-enhanced semantic search."""
    if not CACHED_FAQS or FAQ_EMBEDDINGS is None:
        return "The Knowledge Base is currently empty or unavailable."

    query_embedding = SEARCH_MODEL.encode([query])
    similarities = cosine_similarity(query_embedding, FAQ_EMBEDDINGS)[0]
    
    # Get the indices of the top_k most similar FAQs
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]

    # Format the context for the LLM
    context_parts = []
    for index in top_k_indices:
        # Use a similarity threshold to filter out irrelevant results
        if similarities[index] > 0.35: # This threshold can be tuned
            faq = CACHED_FAQS[index]
            context_parts.append(f"Question: {faq['question']}\nAnswer: {faq['answer']}")
    
    if not context_parts:
        return "No specific information found in the document for this query."
        
    return "\n\n---\n\n".join(context_parts)

# ==============================================================================
# == FLASK CHAT ENDPOINT
# ==============================================================================

@app.route("/chat", methods=["POST"])
def handle_chat():
    """Handles chatbot queries, using MySQL for caching and enhanced knowledge retrieval."""
    data = request.get_json()
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Query is missing from the request"}), 400
    
    if not CACHED_FAQS:
         return jsonify({"error": "Knowledge base is not loaded. Cannot process query."}), 500

    conn = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        # 1. Check for a cached response in the chat_history table
        cursor.execute("SELECT * FROM chat_history WHERE query = %s", (user_query,))
        cached_result = cursor.fetchone()

        if cached_result:
            logger.info(f"Returning cached response for query: '{user_query}'")
            # Safely deserialize JSON fields
            cached_result['keywords'] = json.loads(cached_result.get('keywords') or '[]')
            cached_result['metadata'] = json.loads(cached_result.get('metadata') or '{}')
            return jsonify(cached_result)

        # 2. If not cached, get relevant context using our enhanced semantic search
        relevant_context = get_relevant_context(user_query)

        # 3. Construct the advanced, persona-driven prompt for the LLM
        logger.info(f"Querying main LLM for response and metadata for: '{user_query}'")
        
        # *** KEY IMPROVEMENT: Using the advanced prompt ***
        prompt = f"""
        You are the BioGenex Virtual Assistant, an expert on the company's products, technologies, and operations. Your persona is professional, knowledgeable, and helpful. Your primary goal is to provide accurate and concise answers based ONLY on the context provided.

        **Instructions:**
        1.  **Analyze the Context:** Carefully review the provided "Relevant Context" which contains information from the BioGenex knowledge base. Your answer MUST be derived solely from this text. Do not invent information or use external knowledge.
        2.  **Directly Answer the Question:** Address the user's "Question" directly and accurately.
        3.  **Handle "No Information" Scenarios:** If the context does not contain the information needed to answer the question, state that you cannot find the information in the provided documents and suggest contacting a specific BioGenex department. For example: "I do not have information on that specific topic. For detailed inquiries, I recommend contacting the Technical Support team at support@biogenex.com."
        4.  **Adhere to the JSON Format:** Your entire output must be a single, clean JSON object, without any introductory text like "Here is the JSON object:".

        --- Relevant Context ---
        {relevant_context}
        --- End of Context ---

        --- User's Question ---
        {user_query}
        --- End of Question ---

        Based on the context and the user's question, generate a JSON object with the following exact structure:
        {{
          "response": "A concise and accurate answer to the question based strictly on the provided context. If the information is not in the context, state that and provide a relevant contact email.",
          "keywords": ["A list of 3-5 relevant keywords from the query and context. Examples: 'Xmatrx ELITE', 'IHC', 'Antigen Retrieval', 'Distributor', 'miRNA'."],
          "metadata": {{
            "category": "Classify the query into one of the following categories: 'Corporate Profile', 'Product Information', 'Technical Support', 'Sales & Distribution', or 'Scientific Principles'."
          }}
        }}
        """

        # 4. Query the LLM
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1, # Low temperature for factual, less creative answers
        )
        
        try:
            llm_output = json.loads(completion.choices[0].message.content)
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"Failed to decode or parse JSON from LLM response: {e}")
            return jsonify({"error": "Failed to process the AI model's response."}), 500

        llm_response = llm_output.get("response", "I am sorry, but I could not generate a response.")
        llm_keywords = llm_output.get("keywords", [])
        llm_metadata = llm_output.get("metadata", {})

        # 5. Cache the new response in the database
        cursor.execute(
            "INSERT INTO chat_history (query, response, context, keywords, metadata) VALUES (%s, %s, %s, %s, %s)",
            (
                user_query, 
                llm_response,
                relevant_context,
                json.dumps(llm_keywords),
                json.dumps(llm_metadata)
            )
        )
        conn.commit()
        logger.info(f"Cached new response with metadata for query: '{user_query}'")

        # 6. Return the response to the user
        return jsonify({
            "query": user_query,
            "response": llm_response,
            "context": relevant_context,
            "keywords": llm_keywords,
            "metadata": llm_metadata
        })

    except Error as e:
        logger.error(f"A database error occurred: {e}", exc_info=True)
        return jsonify({"error": "A database error occurred. Please check the logs."}), 500
    except Exception as e:
        logger.error(f"An unexpected error occurred during chat handling: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred. Please check the logs."}), 500
    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
# ==============================================================================
# == MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    init_db()
    load_faqs_from_db()
    # For production, consider using a production-ready WSGI server like Gunicorn or uWSGI
    app.run(host='0.0.0.0', port=5000, debug=True)
