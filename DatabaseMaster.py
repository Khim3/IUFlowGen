import psycopg2
from psycopg2 import sql

class DatabaseMaster:
    def __init__(self, dbname, user, password, host="localhost", port="5432"):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.connection = None

    def connect(self):
        """
        Establish a connection to the PostgreSQL database.
        """
        try:
            self.connection = psycopg2.connect(
                dbname=self.dbname,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            print("Connection to PostgreSQL established successfully.")
        except psycopg2.DatabaseError as error:
            print(f"Error: Unable to connect to the database: {error}")
            self.connection = None

    def close(self):
        """
        Close the connection to the PostgreSQL database.
        """
        if self.connection:
            self.connection.close()
            print("Database connection closed.")
        else:
            print("No active connection to close.")

    def execute_query(self, query, params=None):
        """
        Execute a SQL query and return the result.

        :param query: SQL query to execute.
        :param params: Optional parameters for the SQL query.
        :return: Query result (if any).
        """
        if self.connection:
            with self.connection.cursor() as cursor:
                try:
                    cursor.execute(query, params)
                    self.connection.commit()
                    return cursor.fetchall() if cursor.description else None
                except psycopg2.DatabaseError as error:
                    print(f"Error executing query: {error}")
                    self.connection.rollback()
        else:
            print("No active database connection.")

# Example Usage:

# Initialize the DatabaseMaster with your PostgreSQL credentials for vector_db
db = DatabaseMaster(dbname="vector_db", user="postgres", password="0402")

# Connect to the PostgreSQL database
db.connect()

# Example query (replace with your own SQL query)
query = """
INSERT INTO items (id, embedding) VALUES 
(2, '[0.1,0.2,0.3]'),
(3, '[0.1,0.2,0.3]'),
(4, '[0.1,0.2,0.3]'),
(5, '[0.1,0.2,0.3]'),
(6, '[0.1,0.2,0.3]');
SELECT * FROM items;""" 
result = db.execute_query(query)

if result:
    print(result)

# Close the connection
db.close()
