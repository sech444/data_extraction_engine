import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
import fitz
import cv2
import pytesseract
import numpy as np
from datetime import datetime
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI setup
app = FastAPI()
engine = create_engine('sqlite:///extracted_data.db')
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database model
class DataPoint(Base):
    __tablename__ = 'datapoints'
    id = Column(Integer, primary_key=True, index=True)
    unique_id = Column(String, index=True)
    filing_number = Column(String)
    filing_date = Column(DateTime)
    rcs_number = Column(String)
    dp_value = Column(String)
    dp_unique_value = Column(String)

Base.metadata.create_all(bind=engine)

# PDF processing functions
def extract_data_from_pdf(file_path):
    logger.info(f"Extracting data from PDF: {file_path}")
    pdf_document = fitz.open(file_path)
    data_points = []
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = cv2.imdecode(np.frombuffer(pix.tobytes(), np.uint8), -1)

        anchors = detect_anchors(img)
        data_points.extend(extract_data_points(img, anchors))
    logger.info(f"Extracted data points: {data_points}")
    return data_points

def detect_anchors(img):
    # Example implementation for detecting anchors (this needs to be adjusted for your specific case)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    anchors = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 10 < w < 50 and 10 < h < 50:  # Placeholder for anchor size, adjust as needed
            anchors.append((x, y, w, h))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    logger.info(f"Anchors detected: {anchors}")
    return anchors

def extract_data_points(img, anchors):
    data_points = []
    for anchor in anchors:
        x, y, w, h = anchor
        # Define the relative position of data points to the anchor
        dp_x = x + w + 10  # Placeholder, adjust as needed
        dp_y = y
        dp_w = 100  # Placeholder for data point width, adjust as needed
        dp_h = 30   # Placeholder for data point height, adjust as needed

        # Crop the image around the data point area
        dp_img = img[dp_y:dp_y+dp_h, dp_x:dp_x+dp_w]
        
        # Use Tesseract to extract text from the cropped image
        dp_text = pytesseract.image_to_string(dp_img).strip()
        
        if dp_text:  # Only add data points with meaningful text
            dp = {
                'unique_id': str(uuid.uuid4()),
                'value': dp_text,
                'unique_value': str(uuid.uuid4())
            }
            data_points.append(dp)
    
    logger.info("Data points extracted")
    return data_points

# API endpoints
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...), filing_number: str = Form(...), filing_date: str = Form(...), rcs_number: str = Form(...)):
    contents = await file.read()
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(contents)
    logger.info(f"Received PDF file: {file.filename}")
    
    # Convert filing_date from string to datetime
    try:
        filing_date_dt = datetime.strptime(filing_date, '%d/%m/%Y')
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use DD/MM/YYYY.")
    
    data_points = extract_data_from_pdf(file_path)

    db = SessionLocal()
    try:
        for dp in data_points:
            db_dp = DataPoint(
                unique_id=dp['unique_id'],
                filing_number=filing_number,
                filing_date=filing_date_dt,
                rcs_number=rcs_number,
                dp_value=dp['value'],
                dp_unique_value=dp['unique_value']
            )
            db.add(db_dp)
        db.commit()
        logger.info("Data points committed to the database")
    except Exception as e:
        db.rollback()
        logger.error(f"Error committing data points to the database: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        db.close()

    return {"status": "success"}

@app.get("/get_data/{unique_id}")
async def get_data(unique_id: str):
    db = SessionLocal()
    try:
        data = db.query(DataPoint).filter(DataPoint.unique_id == unique_id).all()
        if not data:
            raise HTTPException(status_code=404, detail="Data not found")
        logger.info(f"Data retrieved for unique_id={unique_id}: {data}")
        return data
    except Exception as e:
        logger.error(f"Error retrieving data for unique_id={unique_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        db.close()

import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

def extract_text_with_ocr(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        if img.shape[2] == 4:  # Convert RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Use Tesseract to extract text
        ocr_text = pytesseract.image_to_string(img)
        text += ocr_text
        logging.debug(f"Extracted OCR text from page {page_num}:\n{ocr_text}")
    
    return text

pdf_path = 'Modification-Type_2A.pdf'
extracted_text = extract_text_with_ocr(pdf_path)
print(extracted_text)
print("-------------------------------------------------------------------")
print("-------------------------------------------------------------------")
print("-------------------------------------------------------------------")
print("-------------------------------------------------------------------")

