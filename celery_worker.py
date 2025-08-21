# celery_worker.py

import multiprocessing
import logging

# Set multiprocessing start method at the very top
try:
    multiprocessing.set_start_method('spawn', force=True)
    logging.info("Set multiprocessing start method to 'spawn'.")
except RuntimeError:
    logging.warning("Multiprocessing start method already set.")

# Import other modules after setting the start method
from dotenv import load_dotenv
import os
import json
from mysql.connector import Error
from celery import Celery
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from groq import Groq
import mysql.connector

load_dotenv()
# from dotenv import env

# --- CELERY CONFIGURATION ---
celery = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# --- SETUP AND CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GROQ_API_KEY = "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" 
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
MODEL_NAME = 'llama3-70b-8192' 

db_config = {
    'host': '195.35.59.66',
    'database': 'u914092009_chat_bot',
    'user': 'u914092009_vidhan',
    'password': '#Vcr@2001'
}

# --- DATABASE LOADING ---
def load_faqs_from_db_for_worker():
    """Loads FAQs and computes embeddings for the worker."""
    conn = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT faq_id, question, answer FROM faqs")
        faqs = cursor.fetchall()
        
        if not faqs:
            logger.warning("No FAQs found in the database for worker.")
            return [], None

        questions = [faq['question'] for faq in faqs]
        # Load model here to compute embeddings, using CPU to avoid CUDA issues
        search_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        embeddings = search_model.encode(questions, show_progress_bar=True)
        logger.info(f"Worker loaded and embedded {len(faqs)} FAQs.")
        return faqs, embeddings
    except Error as e:
        logger.error(f"Worker could not load FAQs: {e}", exc_info=True)
        return [], None
    finally:
        if conn and conn.is_connected():
            conn.close()

# Load FAQs into global variables for the worker process
CACHED_FAQS, FAQ_EMBEDDINGS = load_faqs_from_db_for_worker()

# --- CELERY TASK DEFINITION ---
@celery.task(name='process_chat_query')
def process_chat_query(user_query):
    """
    This is the background task. It performs all the slow operations.
    """
    # Load the model inside the task, using CPU
    logger.info("Loading sentence transformer model for semantic search in task...")
    search_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    logger.info("Sentence transformer model loaded in task.")

    # 1. Get relevant context (Semantic Search)
    if not CACHED_FAQS or FAQ_EMBEDDINGS is None:
        return {"error": "Knowledge base not available in worker."}

    query_embedding = search_model.encode([user_query])
    similarities = cosine_similarity(query_embedding, FAQ_EMBEDDINGS)[0]
    top_k_indices = np.argsort(similarities)[-3:][::-1]
    
    context_parts = []
    for index in top_k_indices:
        if similarities[index] > 0.35:
            faq = CACHED_FAQS[index]
            context_parts.append(f"Question: {faq['question']}\nAnswer: {faq['answer']}")
    
    relevant_context = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant information found."

    # 2. Query the LLM
    logger.info(f"Worker processing LLM for query: '{user_query}'")
    prompt = f"""
    You are a highly intelligent assistant for a company named BioGenex.
    Your task is to answer the user's question based *only* on the provided context below.
    If the context does not contain the answer, state that you cannot find the information.
    Your response MUST be a single, clean JSON object with the following structure:
    {{
      "response": "A concise and helpful answer to the user's question, derived strictly from the context.",
      "keywords": ["A list of 3-5 relevant keywords from the query and context."],
      "metadata": {{ "category": "A suitable category for the query, such as 'Instrumentation', 'Reagents', 'Technical Support', or 'General'." }}
    }}
    --- Relevant Context ---
    {relevant_context}
    --- End of Context ---
    User's Question: {user_query}
    """
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=4096,
            response_format={"type": "json_object"},
        )
        llm_output = json.loads(completion.choices[0].message.content)
    except Exception as e:
        logger.error(f"LLM call failed in worker: {e}")
        return {"error": "Failed to get response from LLM."}

    # 3. Cache the result in the database
    conn = None
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_history (query, response, context, keywords, metadata) VALUES (%s, %s, %s, %s, %s)",
            (
                user_query, 
                llm_output.get("response", ""),
                relevant_context,
                json.dumps(llm_output.get("keywords", [])),
                json.dumps(llm_output.get("metadata", {}))
            )
        )
        conn.commit()
        logger.info(f"Worker cached new response for query: '{user_query}'")
    except Error as e:
        logger.error(f"Worker failed to cache response: {e}")
    finally:
        if conn and conn.is_connected():
            conn.close()

    # 4. Return the final result
    return {
        "query": user_query,
        "response": llm_output.get("response", "No valid response generated."),
        "context": relevant_context,
        "keywords": llm_output.get("keywords", []),
        "metadata": llm_output.get("metadata", {})
    }
