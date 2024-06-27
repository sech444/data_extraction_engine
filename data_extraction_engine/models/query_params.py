# data_extraction_engine/models/query_params.py

from pydantic import BaseModel
from typing import Optional

class QueryParams(BaseModel):
    filing_number: Optional[str] = None
    filing_date: Optional[str] = None
    rcs_number: Optional[str] = None
    dp_value: Optional[str] = None
    dp_unique_value: Optional[str] = None


