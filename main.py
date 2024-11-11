import db
import register_app
import api.openapi as openapi
import app.app as app

if __name__ == "__main__":
    # Ensure that any multiprocessing setup is done properly here
    db.create_schema_and_table()

    # register_app.Initialize()

    openapi.start_api()

    SessionLocal, Asset = db.setup_database()

    # Now, call the function to start the forecast and training processes
    app.background_worker(SessionLocal, Asset)
