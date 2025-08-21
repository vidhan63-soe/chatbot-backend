# app.py (Refactored for Celery)

import os
import json
import logging
from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
from celery import Celery
from celery.result import AsyncResult

# --- FLASK & CELERY APP INITIALIZATION ---
app = Flask(__name__)
CORS(app)

# Configure Celery to use the same broker and backend as the worker
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# --- SETUP AND CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Note: The heavy lifting (model loading, DB access, LLM calls) is now in celery_worker.py
# The Flask app is lightweight and only handles web requests.

# ==============================================================================
# == FLASK ENDPOINTS
# ==============================================================================

@app.route("/chat", methods=["POST"])
def handle_chat():
    """
    Receives a query, queues it as a background task, and returns a task ID.
    This endpoint responds almost instantly.
    """
    data = request.get_json()
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Query is missing"}), 400

    # Offload the processing to the Celery worker
    # We call the task by its registered name 'process_chat_query'
    task = celery.send_task('process_chat_query', args=[user_query])
    
    logger.info(f"Queued task {task.id} for query: '{user_query}'")

    # Return a response that includes the URL to check for the result
    return jsonify({
        "message": "Your query is being processed.",
        "task_id": task.id,
        "result_url": url_for('get_result', task_id=task.id, _external=True)
    }), 202 # 202 Accepted status code is appropriate here

@app.route("/result/<string:task_id>", methods=["GET"])
def get_result(task_id):
    """
    Allows the client to poll for the result of a background task.
    """
    task_result = AsyncResult(task_id, app=celery)

    if task_result.ready():
        if task_result.successful():
            result = task_result.get()
            return jsonify({
                "status": "SUCCESS",
                "data": result
            })
        else:
            # Task failed
            return jsonify({
                "status": "FAILURE",
                "error": str(task_result.info) # Get exception info
            }), 500
    else:
        # Task is still pending
        return jsonify({"status": "PENDING"}), 202

# ==============================================================================
# == MAIN EXECUTION
# ==============================================================================
if __name__ == '__main__':
    # The Flask app no longer needs to load models or DB on startup
    app.run(debug=True, port=5000)
