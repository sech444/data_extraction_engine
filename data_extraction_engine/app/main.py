# data_extraction_engine/app/main.py

from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_, text
from ..utils.pdf_processor import process_pdf, store_data_to_db
from ..models.database import SessionLocal, init_db
from ..models.query_params import QueryParams
from ..models.models import DataPoint
from typing import List, Any

app = FastAPI()

init_db()

# Dependency to get the SQLAlchemy session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post('/submit_pdf')
async def submit_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF file.")
    
    data_points = process_pdf(await file.read())
    store_data_to_db(db, data_points)
    return JSONResponse(content={'status': 'success', 'data_points': data_points})

@app.post('/query_data')
async def query_data(params: QueryParams, db: Session = Depends(get_db)):
    print(params)  # Log the query parameters
    query_conditions = []
    if params.filing_number:
        query_conditions.append(DataPoint.filing_number == params.filing_number)
    if params.filing_date:
        query_conditions.append(DataPoint.filing_date == params.filing_date)
    if params.rcs_number:
        query_conditions.append(DataPoint.rcs_number == params.rcs_number)
    if params.dp_value:
        query_conditions.append(DataPoint.dp_value == params.dp_value)
    if params.dp_unique_value:
        query_conditions.append(DataPoint.dp_unique_value == params.dp_unique_value)

    results = db.query(DataPoint).filter(and_(*query_conditions)).all()
    return JSONResponse(content={'status': 'success', 'results': [result.__dict__ for result in results if '_sa_instance_state' not in result.__dict__]})

@app.post('/query_data_perview')
async def query_data_perview(db: Session = Depends(get_db)):
    query = text("SELECT * FROM data_points")
    cursor = db.execute(query)
    rows: List[Any] = cursor.fetchall()
    for row in rows:
        print(row)  # Replace print with your desired data processing

    row_data = [dict(row) for row in rows]  # Convert SQLAlchemy row objects to dictionaries
    return JSONResponse(content={'status': 'success', 'results': row_data})

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host='0.0.0.0', port=8000)

