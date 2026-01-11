import json
import os
from datetime import datetime

HISTORY_FILE = "history.json"

def save_search(topic, confidence_score, report_html="", x_intel_data=None):
    """
    Appends a search result to the history JSON.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    new_entry = {
        "timestamp": timestamp,
        "topic": topic,
        "confidence_score": confidence_score,
        "report_html": report_html,
        "x_intel_data": x_intel_data
    }
    
    # Load existing history
    history = load_history()
    
    # Append new entry at the beginning
    history.insert(0, new_entry)
    
    # Save back to file
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

def load_history():
    """
    Loads the search history from JSON.
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return []

def get_search_by_index(index):
    """
    Retrieves a specific search entry by its index.
    """
    history = load_history()
    if 0 <= index < len(history):
        return history[index]
    return None
