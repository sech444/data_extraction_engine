# data_extraction_engine/utils/pdf_processor.py

import fitz  # PyMuPDF
import cv2
import pytesseract
import numpy as np
from typing import List, Dict
from sqlalchemy.orm import Session
from ..models import models

def process_pdf(file, threshold=0.8) -> List[Dict]:
    pdf_document = fitz.open(stream=file, filetype="pdf")
    data_points = []

    # Extract templates from the PDF
    templates = extract_templates_from_pdf(pdf_document)
    if not templates:
        raise ValueError("No templates found in the PDF document")

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Pre-process image
        img = preprocess_image(img)

        # Detect anchors
        anchors = detect_anchors(img, templates, threshold)

        for anchor in anchors:
            # Extract DPs using masks
            dp = extract_data_points(img, anchor)
            data_points.append(dp)

            # Detect and extract checkbox states
            checkboxes = detect_checkboxes(img, anchor)
            data_points.extend(checkboxes)

    return data_points

def preprocess_image(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to improve contrast
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply Gaussian filtering for noise reduction
    blur = cv2.GaussianBlur(thresh, (5, 5), 0)

    return blur

def detect_anchors(img, templates, threshold=0.8):
    """
    Detects anchor locations in the image using template matching.

    Args:
        img: The image as a NumPy array.
        templates: A list of anchor template images as NumPy arrays.
        threshold: A float between 0 and 1 (inclusive) representing the minimum match score for a good match (default: 0.8).

    Returns:
        A list of anchor coordinates (tuples of (x, y) for top-left corner).
    """
    anchors = []
    for template in templates:
        # Convert grayscale templates to RGB if necessary
        if len(template.shape) == 2:
            template = cv2.cvtColor(template, cv2.COLOR_GRAY2RGB)
        
        # Ensure img is in RGB format
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Ensure templates and image have compatible color depth and sizes
        if img.shape[2] != template.shape[2] or img.dtype != template.dtype:
            raise ValueError("Template and image have incompatible color depth or data type.")

        # Apply template matching (normalized correlation coefficient)
        result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

        # Find all locations where the match is above the threshold
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= threshold:
            x, y = max_loc
            anchors.append((x, y))

    return anchors

def extract_templates_from_pdf(pdf_document) -> List[np.ndarray]:
    """
    Extracts templates from the PDF document.

    Args:
        pdf_document: The PyMuPDF document object.

    Returns:
        A list of template images as NumPy arrays.
    """
    templates = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Identify if this page is a template
        if is_template_page(img):
            templates.append(img)
    return templates

def is_template_page(img) -> bool:
    """
    Determines if a given image is a template page.

    Args:
        img: The image as a NumPy array.

    Returns:
        True if the image is a template page, False otherwise.
    """
    # Implement logic to identify if a page is a template
    # This could be based on specific markers, content, etc.
    # For example, if a page contains certain keywords or markers
    return True  # Placeholder implementation

def extract_data_points(img, anchor, mask_offset=(0, 0), mask_size=(50, 20)):
    """
    Extracts data points from the image around a detected anchor.

    Args:
        img: The image as a NumPy array.
        anchor: The anchor location as a tuple (x, y) for top-left corner.
        mask_offset: A tuple (x_offset, y_offset) to adjust the mask position relative to the anchor.
        mask_size: A tuple (width, height) defining the size of the mask.

    Returns:
        A dictionary containing the extracted data point information.
    """
    # Adjust anchor position based on mask offset
    anchor_x, anchor_y = anchor
    x, y = anchor_x + mask_offset[0], anchor_y + mask_offset[1]

    # Generate mask around expected data point location
    mask = np.zeros_like(img[:, :, 0])  # Initialize mask as black (all pixels 0)
    mask[y:y+mask_size[1], x:x+mask_size[0]] = 1  # Set mask area to white (all pixels 1)

    # Apply the mask to the image and extract the data point area
    dp_img = cv2.bitwise_and(img, img, mask=mask)

    # Use Tesseract for OCR on the extracted data point image
    text = pytesseract.image_to_string(dp_img, config='--psm 6')

    # Clean and validate the extracted text (optional: consider regular expressions)
    dp_value = clean_text(text)

    # Create a unique ID for the data point
    dp_unique_id = generate_unique_id()

    return {
        'id': dp_unique_id,
        'filing_number': '',  # Replace with actual value from API
        'filing_date': '',  # Replace with actual value from API
        'rcs_number': '',  # Replace with actual value from API
        'dp_value': dp_value,
        'dp_unique_value': dp_unique_id  # Use this for internal differentiation
    }

def detect_checkboxes(img, anchor) -> List[Dict]:
    # Detect checkboxes and extract their state
    checkboxes = []
    return checkboxes

def generate_unique_id() -> int:
    # Generate a unique ID for each DP
    return 1

def store_data_to_db(db: Session, data: List[Dict]):
    for dp in data:
        db_data_point = models.DataPoint(
            filing_number=dp['filing_number'],
            filing_date=dp['filing_date'],
            rcs_number=dp['rcs_number'],
            dp_value=dp['dp_value'],
            dp_unique_value=dp['dp_unique_value']
        )
        print(db_data_point)  # Log each data point being inserted
        db.add(db_data_point)
    db.commit()

def clean_text(text):
    # Implement text cleaning and validation
    return text.strip()
