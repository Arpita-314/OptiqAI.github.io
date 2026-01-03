from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class ExperimentResult(Base):
    __tablename__ = "experiment_results"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    metric = Column(Float)
    notes = Column(String)

# PostgreSQL connection string
DATABASE_URL = "postgresql+psycopg2://foaml_user:yourpassword@localhost/foaml_db"

engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)

