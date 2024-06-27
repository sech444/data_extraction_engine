# # data_extraction_engine/models/database.py

# # import sqlite3
# from sqlalchemy import create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker

# SQLALCHEMY_DATABASE_URL = "sqlite:///./data_extraction_app.db"
# # SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"

# engine = create_engine(
#     SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
# )
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base = declarative_base()

# def init_db():
#     conn = sqlite3.connect('data_points.db')
#     c = conn.cursor()
#     c.execute('''CREATE TABLE IF NOT EXISTS data_points (
#                     id INTEGER PRIMARY KEY,
#                     filing_number TEXT,
#                     filing_date TEXT,
#                     rcs_number TEXT,
#                     dp_value TEXT,
#                     dp_unique_value TEXT)''')
#     conn.commit()
#     conn.close()

# def store_data_to_db(data):
#     conn = sqlite3.connect('data_points.db')
#     c = conn.cursor()
#     for dp in data:
#         c.execute('''INSERT INTO data_points (filing_number, filing_date, rcs_number, dp_value, dp_unique_value) 
#                      VALUES (?, ?, ?, ?, ?)''', 
#                      (dp['filing_number'], dp['filing_date'], dp['rcs_number'], dp['dp_value'], dp['dp_unique_value']))
#     conn.commit()
#     conn.close()

# data_extraction_engine/models/database.py

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./data_extraction_app.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def init_db():
    from .models import DataPoint
    Base.metadata.create_all(bind=engine)
