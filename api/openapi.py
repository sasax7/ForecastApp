from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional  # Import Optional for making fields optional
from sqlalchemy import (
    MetaData,
    Table,
    create_engine,
)
from sqlalchemy.orm import sessionmaker, Session
import os
import subprocess


# Pydantic model for the 'assets_to_forecast' table
class AssetModel(BaseModel):
    id: Optional[int]  # Allow 'id' to be optional for creation
    gai: str
    target_attribute: str
    feature_attributes: Optional[dict]  # JSONB field
    forecast_length: int
    start_date: Optional[str]  # Allow date to be optional
    parameters: Optional[dict]  # JSONB field
    datalength: Optional[int]
    hyperparameters: Optional[dict]  # JSONB field
    trainingparameters: Optional[dict]  # JSONB field
    latest_timestamp: Optional[str]  # Timestamp
    context_length: Optional[int]  # Allow optional
    processing_status: Optional[str]  # Allow optional
    scaler: Optional[bytes]  # Allow optional
    state: Optional[bytes]  # Allow optional

    class Config:
        from_attributes = True  # Updated for Pydantic v2
        orm_mode = True


# Function to create FastAPI app with dynamic DATABASE_URL
def create_api(DATABASE_URL: str):
    # Create the database engine with dynamic DATABASE_URL
    engine = create_engine(DATABASE_URL)

    # Reflect the assets_to_forecast table from the forecast schema
    metadata = MetaData()
    Asset = Table(
        "assets_to_forecast", metadata, autoload_with=engine, schema="forecast"
    )

    # Create a new session for database interactions
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # FastAPI app instance
    app = FastAPI()

    # Dependency to get DB session
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    # API to get all assets
    @app.get("/assets", response_model=list[AssetModel])
    def read_assets(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
        result = db.execute(Asset.select().offset(skip).limit(limit))
        assets = result.fetchall()

        # Convert the rows to dictionaries to match Pydantic model
        assets_dict = [dict(row) for row in assets]

        return assets_dict

    # API to get an asset by ID
    @app.get("/assets/{id}", response_model=AssetModel)
    def read_asset(id: int, db: Session = Depends(get_db)):
        result = db.execute(Asset.select().where(Asset.c.id == id)).first()
        if result is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return result

    @app.get("/assets/gai/{gai}", response_model=AssetModel)
    def read_asset_by_gai(gai: str, db: Session = Depends(get_db)):
        result = db.execute(Asset.select().where(Asset.c.gai == gai)).first()
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"Asset with GAI {gai} not found"
            )
        return result

    @app.get("/assets/search", response_model=AssetModel)
    def read_asset_by_gai_target_forecast(
        gai: str,
        target_attribute: str,
        forecast_length: int,
        db: Session = Depends(get_db),
    ):
        result = db.execute(
            Asset.select()
            .where(Asset.c.gai == gai)
            .where(Asset.c.target_attribute == target_attribute)
            .where(Asset.c.forecast_length == forecast_length)
        ).first()

        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Asset with GAI {gai}, target attribute {target_attribute}, and forecast length {forecast_length} not found",
            )

        return result

    # API to create a new asset
    @app.post("/assets", response_model=AssetModel)
    def create_asset(asset: AssetModel, db: Session = Depends(get_db)):
        # Insert new asset data
        new_asset = Asset.insert().values(
            gai=asset.gai,
            target_attribute=asset.target_attribute,
            feature_attributes=asset.feature_attributes,
            forecast_length=asset.forecast_length,
            start_date=asset.start_date,
            parameters=asset.parameters,
            datalength=asset.datalength,
            hyperparameters=asset.hyperparameters,
            trainingparameters=asset.trainingparameters,
            latest_timestamp=asset.latest_timestamp,
            context_length=asset.context_length,
            processing_status=asset.processing_status or "new",
            scaler=asset.scaler,
            state=asset.state,
        )
        db.execute(new_asset)
        db.commit()
        return asset

    # API to update an existing asset
    @app.put("/assets/{id}", response_model=AssetModel)
    def update_asset(id: int, asset: AssetModel, db: Session = Depends(get_db)):
        db_asset = db.execute(Asset.select().where(Asset.c.id == id)).first()
        if db_asset is None:
            raise HTTPException(status_code=404, detail="Asset not found")

        update_query = (
            Asset.update()
            .where(Asset.c.id == id)
            .values(
                gai=asset.gai,
                target_attribute=asset.target_attribute,
                feature_attributes=asset.feature_attributes,
                forecast_length=asset.forecast_length,
                start_date=asset.start_date,
                parameters=asset.parameters,
                datalength=asset.datalength,
                hyperparameters=asset.hyperparameters,
                trainingparameters=asset.trainingparameters,
                latest_timestamp=asset.latest_timestamp,
                context_length=asset.context_length,
                processing_status=asset.processing_status,
                scaler=asset.scaler,
                state=asset.state,
            )
        )
        db.execute(update_query)
        db.commit()
        return asset

    # API to delete an asset by ID
    @app.delete("/assets/{id}", response_model=AssetModel)
    def delete_asset(id: int, db: Session = Depends(get_db)):
        db_asset = db.execute(Asset.select().where(Asset.c.id == id)).first()
        if db_asset is None:
            raise HTTPException(status_code=404, detail="Asset not found")

        delete_query = Asset.delete().where(Asset.c.id == id)
        db.execute(delete_query)
        db.commit()
        return db_asset

    return app  # Return the app instance without calling `uvicorn.run()`


def start_api():
    python_executable = os.path.join(".venv", "Scripts", "python.exe")
    port = os.getenv("API_SERVER_PORT")
    subprocess.Popen(
        [
            python_executable,
            "-m",
            "uvicorn",
            "openapi:fastapi_app",
            "--host",
            "0.0.0.0",
            "--port",
            port,
            "--reload",
        ]
    )
