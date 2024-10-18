import logging
from sqlalchemy.exc import IntegrityError

# Set up logging
logging.basicConfig(level=logging.INFO)

from sqlalchemy import Table, Column, Integer, String, JSON, Date, TIMESTAMP, MetaData


# Define the Asset table using SQLAlchemy
def create_asset_table(metadata, engine):
    Asset = Table(
        "assets_to_forecast",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("gai", String(255), nullable=False),
        Column("target_attribute", String(255), nullable=False),
        Column("feature_attributes", JSON),
        Column("forecast_length", Integer, nullable=False),
        Column("start_date", Date),
        Column("parameters", JSON),
        Column("hyperparameters", JSON),
        Column("latest_timestamp", TIMESTAMP),
        Column("context_length", Integer),
        schema="forecast",
        autoload_with=engine,
    )
    return Asset


# Function to get all assets
def get_all_assets(SessionLocal, Asset, skip: int = 0, limit: int = 1000):
    logging.info(f"Fetching all assets (skip: {skip}, limit: {limit})")

    with SessionLocal() as session:
        query = Asset.select().offset(skip).limit(limit)
        result = session.execute(query)

        # Fetch all results and convert each row to a dictionary
        assets = result.fetchall()
        assets_dict = [
            dict(row._mapping) for row in assets
        ]  # Use row._mapping to convert row to dict

        return assets_dict


# Function to get an asset by ID
def get_asset_by_id(SessionLocal, Asset, id: int):
    logging.info(f"Fetching asset with ID {id}")

    with SessionLocal() as session:
        query = Asset.select().where(Asset.c.id == id)
        result = session.execute(query).first()
        if result is None:
            raise ValueError(f"Asset with ID {id} not found.")
        return result


# Function to get an asset by GAI
def get_asset_by_gai(SessionLocal, Asset, gai: str):
    logging.info(f"Fetching asset with GAI {gai}")

    with SessionLocal() as session:
        query = Asset.select().where(Asset.c.gai == gai)
        result = session.execute(query).first()
        if result is None:
            raise ValueError(f"Asset with GAI {gai} not found.")
        return result


# Function to get an asset by GAI, target_attribute, and forecast_length
def get_asset_by_gai_target_forecast(
    SessionLocal, Asset, gai: str, target_attribute: str, forecast_length: int
):
    logging.info(
        f"Fetching asset with GAI {gai}, target attribute {target_attribute}, and forecast length {forecast_length}"
    )

    with SessionLocal() as session:
        query = (
            Asset.select()
            .where(Asset.c.gai == gai)
            .where(Asset.c.target_attribute == target_attribute)
            .where(Asset.c.forecast_length == forecast_length)
        )
        result = session.execute(query).first()
        if result is None:
            raise ValueError(
                f"Asset with GAI {gai}, target attribute {target_attribute}, and forecast length {forecast_length} not found."
            )
        return result


def create_asset(
    SessionLocal,
    Asset,
    gai: str,
    target_attribute: str,
    forecast_length: int,
    feature_attributes: dict = None,
    start_date: str = None,
    parameters: dict = None,
    hyperparameters: dict = None,
    latest_timestamp: str = None,
    context_length: int = None,
):
    logging.info(
        f"Creating new asset with GAI {gai}, target attribute {target_attribute}, and forecast length {forecast_length}"
    )

    # Insert a new asset with the provided values
    new_asset = Asset.insert().values(
        gai=gai,
        target_attribute=target_attribute,
        feature_attributes=feature_attributes,
        forecast_length=forecast_length,
        start_date=start_date,
        parameters=parameters,
        hyperparameters=hyperparameters,
        latest_timestamp=latest_timestamp,
        context_length=context_length,
    )

    with SessionLocal() as session:
        try:
            session.execute(new_asset)
            session.commit()
            logging.info("Asset created successfully.")
        except IntegrityError:
            # Catch unique constraint violation and log that the asset already exists
            session.rollback()  # Roll back the session to prevent further issues
            logging.info(f"Asset with GAI {gai} already exists. Skipping creation.")


# Function to update an existing asset
def update_asset(
    SessionLocal,
    Asset,
    id: int,
    gai: str,
    target_attribute: str,
    forecast_length: int,
    feature_attributes: dict = None,
    start_date: str = None,
    parameters: dict = None,
    hyperparameters: dict = None,
    latest_timestamp: str = None,
    context_length: int = None,
):
    logging.info(f"Updating asset with ID {id}")

    with SessionLocal() as session:
        db_asset = session.execute(Asset.select().where(Asset.c.id == id)).first()
        if db_asset is None:
            raise ValueError(f"Asset with ID {id} not found.")

        # Update the asset with the new values
        update_query = (
            Asset.update()
            .where(Asset.c.id == id)
            .values(
                gai=gai,
                target_attribute=target_attribute,
                feature_attributes=feature_attributes,
                forecast_length=forecast_length,
                start_date=start_date,
                parameters=parameters,
                hyperparameters=hyperparameters,
                latest_timestamp=latest_timestamp,
                context_length=context_length,
            )
        )

        session.execute(update_query)
        session.commit()
        logging.info(f"Asset with ID {id} updated successfully.")


# Function to delete an asset by ID
def delete_asset(SessionLocal, Asset, id: int):
    logging.info(f"Deleting asset with ID {id}")

    with SessionLocal() as session:
        db_asset = session.execute(Asset.select().where(Asset.c.id == id)).first()
        if db_asset is None:
            raise ValueError(f"Asset with ID {id} not found.")

        delete_query = Asset.delete().where(Asset.c.id == id)
        session.execute(delete_query)
        session.commit()
        logging.info(f"Asset with ID {id} deleted successfully.")
