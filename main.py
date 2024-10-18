import db
import register_app
import api.openapi as openapi
import app.app as app


db.create_schema_and_table()

register_app.Initialize()

openapi.start_api()

SessionLocal, Asset = db.setup_database()

app.start_forecast(SessionLocal, Asset)
