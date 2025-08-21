# app.py (Updated to sync with Celery for asynchronous processing)

import os
import json
import logging
import mysql.connector
from mysql.connector import Error

from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
from celery.result import AsyncResult

# --- Celery Integration ---
# Import the process_chat_query task from your worker file.
# This allows the Flask app to send jobs to the Celery worker.
from celery_worker import process_chat_query

# --- SETUP AND CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- DATABASE CONFIGURATION ---
# This remains as the app still needs to check the cache and the worker needs it too.
db_config = {
    'host': '195.35.59.66',
    'database': 'u914092009_chat_bot',
    'user': 'u914092009_vidhan',
    'password': '#Vcr@2001'
}

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)
CORS(app)

# ==============================================================================
# == IMPORTANT NOTE ON REMOVED CODE
# ==============================================================================
# The following components have been REMOVED from app.py because their
# responsibilities are now handled entirely by the Celery worker:
#
# 1. SentenceTransformer Model Loading (SEARCH_MODEL): The worker loads its own instance.
# 2. In-Memory FAQ Cache (CACHED_FAQS, FAQ_EMBEDDINGS): The worker manages this.
# 3. load_faqs_from_db(): The worker loads the knowledge base on startup.
# 4. get_relevant_context(): The entire semantic search logic is now in the worker.
# 5. Groq LLM Client: The LLM is now queried only by the worker.
#
# This makes the Flask app lightweight and focused on handling web requests.
# ==============================================================================


# ==============================================================================
# == DATABASE INITIALIZATION
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
        conn.commit()
        logger.info(f"Database connection to '{db_config['database']}' successful and tables are ready.")
    except Error as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()

# ==============================================================================
# == FLASK ENDPOINTS
# ==============================================================================

@app.route("/chat", methods=["POST"])
def handle_chat():
    """
    Handles chatbot queries.
    1. Checks for a cached response for an instant reply.
    2. If not cached, dispatches the query to a Celery worker for background processing.
    """
    data = request.get_json()
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Query is missing from the request"}), 400
    
    conn = None
    try:
        # 1. Check for a cached response in the database (this is a fast operation)
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM chat_history WHERE query = %s", (user_query,))
        cached_result = cursor.fetchone()

        if cached_result:
            logger.info(f"Returning cached response for query: '{user_query}'")
            # Safely deserialize JSON fields before returning
            cached_result['keywords'] = json.loads(cached_result.get('keywords') or '[]')
            cached_result['metadata'] = json.loads(cached_result.get('metadata') or '{}')
            return jsonify({"status": "SUCCESS", "data": cached_result})

        # 2. If not cached, send the task to the Celery worker
        logger.info(f"No cache hit. Dispatching query to Celery worker: '{user_query}'")
        task = process_chat_query.delay(user_query)
        
        # 3. Respond immediately with a task ID for the client to track
        # The HTTP 202 Accepted status code indicates the request has been accepted
        # for processing, but the processing has not been completed.
        return jsonify({
            "status": "PENDING",
            "task_id": task.id,
            "check_status_url": url_for('get_task_status', task_id=task.id, _external=True)
        }), 202

    except Error as e:
        logger.error(f"A database error occurred: {e}", exc_info=True)
        return jsonify({"error": "A database error occurred. Please check the logs."}), 500
    except Exception as e:
        logger.error(f"An unexpected error occurred during chat handling: {e}", exc_info=True)
        return jsonify({"error": "An internal server error occurred. Please check the logs."}), 500
    finally:
        if conn and conn.is_connected():
            conn.close()

@app.route("/result/<string:task_id>", methods=["GET"])
def get_task_status(task_id: str):
    """
    Allows the client to poll for the result of a background task
    using the task_id provided by the /chat endpoint.
    """
    # Use AsyncResult to check the state of the task
    task_result = AsyncResult(task_id)
    
    if task_result.ready():
        if task_result.successful():
            # The task completed successfully
            result = task_result.get()
            return jsonify({"status": "SUCCESS", "data": result})
        else:
            # The task failed
            return jsonify({"status": "FAILURE", "error": str(task_result.info)}), 500
    else:
        # The task is still being processed
        return jsonify({"status": "PENDING"}), 202

# ==============================================================================
# == MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    init_db()
    # Note: load_faqs_from_db() is no longer called here.
    # For production, use a production-ready WSGI server like Gunicorn.
    app.run(host='0.0.0.0', port=5000, debug=True)
