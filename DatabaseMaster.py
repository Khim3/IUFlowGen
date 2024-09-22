import psycopg2
from psycopg2 import sql

class DatabaseMaster:
    def __init__(self, dbname, user, password, host="localhost", port="5432"):
        """
        Initialize the DatabaseMaster with connection parameters.
        
        :param dbname: Name of the database to connect to.
        :param user: Username used to authenticate.
        :param password: Password used to authenticate.
        :param host: Database host address (defaults to localhost).
        :param port: Connection port number (defaults to 5432).
        """
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
query = "SELECT * FROM items;"  # Adjust based on your table name and structure
result = db.execute_query(query)

if result:
    print(result)

# Close the connection
db.close()
