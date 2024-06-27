# data_extraction_engine/models/models.py

from sqlalchemy import Column, Integer, String
from .database import Base

class DataPoint(Base):
    __tablename__ = "data_points"

    id = Column(Integer, primary_key=True, index=True)
    filing_number = Column(String, index=True)
    filing_date = Column(String, index=True)
    rcs_number = Column(String, index=True)
    dp_value = Column(String, index=True)
    dp_unique_value = Column(String, index=True)
