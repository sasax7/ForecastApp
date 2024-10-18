import psycopg2
from psycopg2 import OperationalError
from config import db_url, db_url_sql

from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
)
from sqlalchemy.orm import sessionmaker


# Function to set up the database engine and session
def setup_database():
    # Create the database engine
    DATABASE_URL = db_url_sql
    engine = create_engine(DATABASE_URL)

    # Use MetaData to reflect the 'assets_to_forecast' table from the 'forecast' schema
    metadata = MetaData()
    Asset = Table(
        "assets_to_forecast", metadata, autoload_with=engine, schema="forecast"
    )

    # Create a new session for database interactions
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    return SessionLocal, Asset


def create_schema_and_table():
    try:
        # Establish connection to the PostgreSQL database
        connection = psycopg2.connect(db_url)
        cursor = connection.cursor()

        # Step 1: Create the 'forecast' schema if it doesn't exist
        create_schema_query = "CREATE SCHEMA IF NOT EXISTS forecast;"
        cursor.execute(create_schema_query)
        print("Schema 'forecast' created (or already exists).")

        # Step 2: Create the 'assets' table in the 'forecast' schema
        create_table_query = """
        CREATE TABLE IF NOT EXISTS forecast.assets_to_forecast (
            id SERIAL PRIMARY KEY,  
            gai VARCHAR(255) NOT NULL,
            target_attribute VARCHAR(255) NOT NULL,
            feature_attributes JSONB,
            forecast_length INT NOT NULL,
            start_date DATE,
            parameters JSONB,
            hyperparameters JSONB,
            latest_timestamp TIMESTAMP,
            context_length INT,
            UNIQUE(gai, target_attribute, forecast_length)  -- Enforce uniqueness of the combination
        );
        """
        cursor.execute(create_table_query)
        print("Table 'assets' created (or already exists).")

        # Commit the changes to the database
        connection.commit()

        # Close the cursor and connection
        cursor.close()
        connection.close()

    except OperationalError as e:
        print(f"Connection failed: {e}")


# def check_and_create_forecast_app(db_url):
#     try:
#         # Establish connection to the PostgreSQL database
#         connection = psycopg2.connect(db_url)
#         cursor = connection.cursor()

#         # Check if the 'forecast' app already exists in eliona_app table
#         cursor.execute(
#             "SELECT app_name FROM eliona_app WHERE app_name = %s", ("forecast",)
#         )
#         app = cursor.fetchone()

#         if app:
#             print("App 'forecast' already exists.")
#         else:
#             print("App 'forecast' does not exist. Checking eliona_store...")

#             # Check if the 'forecast' entry exists in eliona_store table
#             cursor.execute(
#                 "SELECT app_name FROM eliona_store WHERE app_name = %s", ("forecast",)
#             )
#             store_entry = cursor.fetchone()

#             if not store_entry:
#                 print(
#                     "App 'forecast' does not exist in eliona_store. Creating new entry..."
#                 )

#                 # Insert a new 'forecast' entry into eliona_store with appropriate values
#                 insert_store_query = """
#                 INSERT INTO eliona_store (app_name, category, version, metadata, created_at)
#                 VALUES (%s, %s, %s, %s, now())
#                 """

#                 # Update the category with a valid value
#                 cursor.execute(
#                     insert_store_query,
#                     (
#                         "forecast",  # app_name
#                         "app",  # Replace with valid category
#                         "1.0",  # version (example value)
#                         "{}",  # metadata (default empty JSON)
#                     ),
#                 )

#                 connection.commit()
#                 print("App 'forecast' added to eliona_store.")

#             # Now insert the app into eliona_app
#             insert_app_query = """
#             INSERT INTO eliona_app (app_name, enable, created_at)
#             VALUES (%s, %s, now())
#             """
#             cursor.execute(insert_app_query, ("forecast", True))

#             # Commit the changes to the database
#             connection.commit()

#             print("App 'forecast' created successfully in eliona_app.")

#             # Now create schema and table for forecast
#             create_schema_and_table(db_url)

#         # Close the cursor and connection
#         cursor.close()
#         connection.close()

#     except OperationalError as e:
#         print(f"Connection failed: {e}")


# if __name__ == "__main__":
#     db_url = os.getenv("CONNECTION_STRING")
#     check_and_create_forecast_app(db_url)
