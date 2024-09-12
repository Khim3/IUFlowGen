import psycopg2
from psycopg2 import sql

class DatabaseMaster:
    def __init__(self, dbname, user, password, host='localhost', port='5432'):
        """Initialize the connection to the PostgreSQL database."""
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.conn = None
        self.cursor = None
        self.connect()

    def connect(self):
        """Establish a connection to the database."""
        try:
            self.conn = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            self.cursor = self.conn.cursor()
            print("Connection to database established.")
        except Exception as e:
            print(f"Error connecting to database: {e}")

    def execute_query(self, query, params=None):
        """Execute a SQL query."""
        try:
            self.cursor.execute(query, params)
            self.conn.commit()
            print("Query executed successfully.")
        except Exception as e:
            print(f"Error executing query: {e}")

    def fetch_results(self):
        """Fetch all results from the last query."""
        try:
            return self.cursor.fetchall()
        except Exception as e:
            print(f"Error fetching results: {e}")
            return []

    def close(self):
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("Connection closed.")

# Example usage
if __name__ == "__main__":
    # Replace with your actual database credentials
    db = DatabaseMaster(dbname='vector_test_db', user='postgres', password='yourpassword')

    # Example: Insert data
    insert_query = "INSERT INTO test_vectors (name, embedding) VALUES (%s, %s);"
    db.execute_query(insert_query, ('Vector4', '[10.0, 11.0, 12.0]'))

    # Example: Select data
    select_query = "SELECT * FROM test_vectors;"
    db.execute_query(select_query)
    results = db.fetch_results()
    for row in results:
        print(row)

    # Close the connection
    db.close()




