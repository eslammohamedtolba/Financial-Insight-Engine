import psycopg
from settings import settings

def get_db_connection(autocommit=False):
    """
    Loads database configuration from environment variables, constructs the connection URL,
    and establishes a connection to the PostgreSQL database using psycopg.
    """
    # The database URL accessed directly from the settings object.
    database_url = str(settings.database_url)
    
    # Return the connection object
    return psycopg.connect(database_url, autocommit=autocommit)

def delete_conversation(thread_id: str):
    """
    Deletes a conversation history from both the 'checkpoints' and 'checkpoint_writes' tables
    to ensure complete removal.
    """
    try:
        # 'with' handles opening and closing the connection
        with get_db_connection() as conn:
            # 'with' also handles cursor management and transactions
            with conn.cursor() as cursor:
                
                # --- Delete from the 'checkpoint_writes' table first ---
                cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'checkpoint_writes');")
                writes_table_exists = cursor.fetchone()[0]
                
                if writes_table_exists:
                    cursor.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (str(thread_id),))

                # --- Delete from the 'checkpoints' table ---
                cursor.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'checkpoints');")
                checkpoints_table_exists = cursor.fetchone()[0]

                if checkpoints_table_exists:
                    cursor.execute("DELETE FROM checkpoints WHERE thread_id = %s", (str(thread_id),))

                if writes_table_exists or checkpoints_table_exists:
                    conn.commit()

    except psycopg.Error as e:
        raise RuntimeError(f"Failed to delete conversation {thread_id} due to a database error.") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while deleting conversation {thread_id}.") from e
