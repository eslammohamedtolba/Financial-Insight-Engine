import sqlite3
import os

# Define the database path consistently with graph.py
DB_DIR = "Data"
DB_FILE = os.path.join(DB_DIR, "graph_memory.sqlite")

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    # Ensure the directory for the database exists
    os.makedirs(DB_DIR, exist_ok=True) 
    return sqlite3.connect(DB_FILE, check_same_thread=False)

def delete_conversation(thread_id: str):
    """
    Deletes a conversation history for a given thread_id from the 'checkpoints' table.
    
    This function first checks if the 'checkpoints' table exists before attempting to 
    delete from it, preventing errors for new users who haven't started a conversation yet.
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if the 'checkpoints' table exists in the database.
            # We query the sqlite_master table which contains metadata about the database schema.
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'")
            
            # fetchone() will return None if the query finds no matching table.
            if cursor.fetchone() is None:
                print(f"Table 'checkpoints' not found. No history to delete for thread_id: {thread_id}")
                return

            # If the table exists, proceed with deleting the history for the given thread_id.
            print(f"Found 'checkpoints' table. Deleting history for thread_id: {thread_id}")
            # The thread_id is passed as a parameter to prevent SQL injection.
            cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (str(thread_id),))
            conn.commit()
            print("History deleted successfully.")

    except sqlite3.Error as e:
        # Catch any other potential SQLite errors for robust error logging.
        print(f"An error occurred while trying to delete conversation history: {e}")