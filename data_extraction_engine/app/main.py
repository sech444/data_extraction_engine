# data_extraction_engine/app/main.py

from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List
import os
from ..models.database import SessionLocal, init_db
from ..models.models import DataPoint
from ..utils.pdf_processor import convert_pdf_to_txt, extract_data_points

app = FastAPI()

init_db()

# Dependency to get the SQLAlchemy session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def store_data_to_db(db: Session, data: List[dict]):
    for dp in data:
        db_data_point = DataPoint(
            filing_number=dp["filing_number"],
            filing_date=dp["filing_date"],
            rcs_number=dp["rcs_number"],
            dp_value=dp["dp_value"],
            dp_unique_value=dp["dp_unique_value"]
        )
        db.add(db_data_point)
    db.commit()

def cleanup_temp_file(file_path: str):
    """Delete the temporary file."""
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted temporary file: {file_path}")
    else:
        print(f"File not found: {file_path}")

@app.post("/submit_pdf")
async def submit_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF file.")
    
    temp_file_path = "temp.pdf"
    contents = await file.read()
    with open(temp_file_path, "wb") as f:
        f.write(contents)
    
    try:
        text = convert_pdf_to_txt(temp_file_path)
        data_points = extract_data_points(text)
        store_data_to_db(db, data_points)
        return JSONResponse(content={"status": "success", "data_points": data_points})
    finally:
        # Ensure the temporary file is cleaned up
        cleanup_temp_file(temp_file_path)

@app.post("/query_data")
async def query_data(filing_number: str = None, filing_date: str = None, rcs_number: str = None, dp_value: str = None, dp_unique_value: str = None, db: Session = Depends(get_db)):
    query = db.query(DataPoint)
    
    if filing_number:
        query = query.filter(DataPoint.filing_number == filing_number)
    if filing_date:
        query = query.filter(DataPoint.filing_date == filing_date)
    if rcs_number:
        query = query.filter(DataPoint.rcs_number == rcs_number)
    if dp_value:
        query = query.filter(DataPoint.dp_value == dp_value)
    if dp_unique_value:
        query = query.filter(DataPoint.dp_unique_value == dp_unique_value)
    
    results = query.all()
    return JSONResponse(content={"status": "success", "results": [result.to_dict() for result in results]})

if __name__ == "__main__":
    # import uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from sqlalchemy.orm import Session
# from typing import List
# import os
# import logging
# from ..models.database import SessionLocal, init_db
# from ..models.models import DataPoint
# from ..utils.pdf_processor import preprocess_pdf, extract_filing_info, locate_sections, extract_data_points_from_sections

# logging.basicConfig(level=logging.DEBUG)

# app = FastAPI()

# init_db()

# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# def store_data_to_db(db: Session, data: List[dict]):
#     for dp in data:
#         db_data_point = DataPoint(
#             filing_number=dp["filing_number"],
#             filing_date=dp["filing_date"],
#             rcs_number=dp["rcs_number"],
#             dp_value=dp["dp_value"],
#             dp_unique_value=dp["dp_unique_value"]
#         )
#         db.add(db_data_point)
#     db.commit()

# def cleanup_temp_file(file_path: str):
#     if os.path.exists(file_path):
#         os.remove(file_path)
#         print(f"Deleted temporary file: {file_path}")
#     else:
#         print(f"File not found: {file_path}")

# @app.post("/submit_pdf")
# async def submit_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
#     if file.content_type != "application/pdf":
#         raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF file.")
    
#     temp_file_path = "temp.pdf"
#     processed_file_path = None  # Initialize processed_file_path

#     try:
#         contents = await file.read()
#         with open(temp_file_path, "wb") as f:
#             f.write(contents)
        
#         logging.debug(f"Uploaded PDF saved at: {temp_file_path}")
        
#         processed_file_path = preprocess_pdf(temp_file_path)
#         filing_info = extract_filing_info(processed_file_path)
#         sections = locate_sections(processed_file_path, filing_info)
#         data_points = extract_data_points_from_sections(processed_file_path, sections)
#         store_data_to_db(db, data_points)
        
#         logging.debug(f"Extracted data points: {data_points}")
        
#         return JSONResponse(content={"status": "success", "data_points": data_points})
#     finally:
#         cleanup_temp_file(temp_file_path)
#         if processed_file_path:
#             cleanup_temp_file(processed_file_path)


# @app.post("/query_data")
# async def query_data(filing_number: str = None, filing_date: str = None, rcs_number: str = None, dp_value: str = None, dp_unique_value: str = None, db: Session = Depends(get_db)):
#     query = db.query(DataPoint)
    
#     if filing_number:
#         query = query.filter(DataPoint.filing_number == filing_number)
#     if filing_date:
#         query = query.filter(DataPoint.filing_date == filing_date)
#     if rcs_number:
#         query = query.filter(DataPoint.rcs_number == rcs_number)
#     if dp_value:
#         query = query.filter(DataPoint.dp_value == dp_value)
#     if dp_unique_value:
#         query = query.filter(DataPoint.dp_unique_value == dp_unique_value)
    
#     results = query.all()
#     return JSONResponse(content={"status": "success", "results": [result.to_dict() for result in results]})

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
