# # data_extraction_engine/utils/pdf_processor.py

import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
import re
import logging
from pytesseract import Output

logging.basicConfig(level=logging.DEBUG)

def preprocess_pdf(pdf_path):
    document = fitz.open(pdf_path)
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        pix = page.get_pixmap()

        # Convert to numpy array for OpenCV processing
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # Fix rotation, zoom, and enhance quality
        img = fix_rotation(img)
        img = enhance_quality(img)

        # Convert back to RGB if needed
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Convert back to bytes
        img_bytes = cv2.imencode('.png', img)[1].tobytes()

        # Create Pixmap from bytes
        processed_pix = fitz.Pixmap(fitz.open("pdf", img_bytes).get_page_pixmap(0))

        # Insert the processed Pixmap back to the PDF
        page.insert_image(page.rect, pixmap=processed_pix)

    processed_path = "processed_" + pdf_path
    document.save(processed_path)
    logging.debug(f"Processed PDF saved at: {processed_path}")
    return processed_path

def fix_rotation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    osd = pytesseract.image_to_osd(gray)
    rotation_angle = extract_rotation_angle(osd)
    rotated_image = rotate_image(image, -rotation_angle)
    logging.debug(f"Rotation angle: {rotation_angle}")
    return rotated_image

def extract_rotation_angle(osd_output):
    angle = 0
    for line in osd_output.split('\n'):
        if 'Rotate:' in line:
            angle = int(line.split(':')[1].strip())
            break
    return angle

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def enhance_quality(image):
    sharpened_image = sharpen_image(image)
    denoised_image = reduce_noise(sharpened_image)
    return denoised_image

def sharpen_image(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def reduce_noise(image):
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised

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

def extract_filing_info(path):
    document = fitz.open(path)
    first_page = document.load_page(0)
    pix = first_page.get_pixmap()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    
    if img.shape[2] == 4:  # Convert RGBA to RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    # Use Tesseract to extract text
    ocr_text = pytesseract.image_to_string(img)
    logging.debug(f"Extracted OCR text from first page:\n{ocr_text}")
    
    filing_info = parse_filing_info(ocr_text)
    logging.debug(f"Parsed filing info: {filing_info}")
    return filing_info

def parse_filing_info(text):
    filing_info = {"sections": []}
    lines = text.split("\n")
    for line in lines:
        if "Section" in line:
            section_number = re.search(r'Section (\d+)', line).group(1)
            anchor = re.search(r'Anchor: (\w+)', line).group(1)
            filing_info["sections"].append({"number": section_number, "anchor": anchor})
    return filing_info

def locate_sections(path, filing_info):
    document = fitz.open(path)
    sections = []
    for page_num in range(1, len(document)):
        page = document.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        if img.shape[2] == 4:  # Convert RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Use Tesseract to extract text data with bounding boxes
        d = pytesseract.image_to_data(img, output_type=Output.DICT)
        n_boxes = len(d['text'])
        
        for i in range(n_boxes):
            if int(d['conf'][i]) > 60:
                text = d['text'][i]
                for section in filing_info["sections"]:
                    if section["anchor"] in text:
                        sections.append((page_num, section["number"]))
                        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Save or display the image with highlighted sections (for verification)
        cv2.imwrite(f'processed_page_{page_num}.png', img)

    logging.debug(f"Located sections: {sections}")
    return sections

def extract_data_points_from_sections(path, sections):
    document = fitz.open(path)
    data_points = []
    for page_num, section_num in sections:
        page = document.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        
        if img.shape[2] == 4:  # Convert RGBA to RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Use Tesseract to extract text
        text = pytesseract.image_to_string(img)
        logging.debug(f"Extracted text from section {section_num} on page {page_num}:\n{text}")
        
        data_points.extend(extract_data_points_from_text(text, section_num))

    return data_points

def extract_data_points_from_text(text, section_num):
    data_points = []
    dp_pattern = r'DP_\d{3}'
    matches = re.findall(dp_pattern, text)
    logging.debug(f"Found matches in section {section_num}: {matches}")
    for match in matches:
        dp_unique_id = generate_unique_id()
        data_points.append({
            'filing_number': '',
            'filing_date': '',
            'rcs_number': '',
            'dp_value': match,
            'dp_unique_value': dp_unique_id,
            'section_number': section_num
        })
    return data_points

def generate_unique_id():
    return np.random.randint(100000, 999999)

def convert_pdf_to_txt(path, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)
    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)
    infile = open(path, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close()
    return text

# # Example usage
# pdf_path = 'Modification-Type_2A.pdf'
# preprocessed_pdf_path = preprocess_pdf(pdf_path)

# # Extract filing information from the first page
# filing_info = extract_filing_info(preprocessed_pdf_path)

# # Locate sections based on tick boxes
# sections = locate_sections(preprocessed_pdf_path, filing_info)

# # Extract data points from the identified sections
# data_points = extract_data_points_from_sections(preprocessed_pdf_path, sections)

# print("Extracted data points:", data_points)


from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import re
import numpy as np

def convert_pdf_to_txt(path, pages=None):
    if not pages:
        pagenums = set()
    else:
        pagenums = set(pages)
    output = StringIO()
    manager = PDFResourceManager()
    converter = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)

    infile = open(path, 'rb')
    for page in PDFPage.get_pages(infile, pagenums):
        interpreter.process_page(page)
    infile.close()
    converter.close()
    text = output.getvalue()
    output.close()
    return text

def extract_data_points(text) -> list:
    """
    Extracts data points from the extracted text.
    Args:
        text: The extracted text from the PDF.
    Returns:
        A list of dictionaries containing the extracted data points.
    """
    data_points = []
    
    # Define the DP pattern (assuming DP_ followed by three digits)
    dp_pattern = r'DP_\d{3}'
    
    # Find all occurrences of the DP pattern in the extracted text
    matches = re.findall(dp_pattern, text)
    
    for match in matches:
        dp_unique_id = generate_unique_id()
        data_points.append({
            'filing_number': '',  # Replace with actual value from API
            'filing_date': '',    # Replace with actual value from API
            'rcs_number': '',     # Replace with actual value from API
            'dp_value': match,
            'dp_unique_value': dp_unique_id
        })
    
    return data_points

def generate_unique_id() -> int:
    # Generate a unique ID for each DP
    return np.random.randint(100000, 999999)


# # new code
# import cv2
# import pytesseract
# from pytesseract import Output
# img = cv2.imread('FB_IMG_1579511433047.jpg')
# d = pytesseract.image_to_data(img, output_type=Output.DICT)
# print(d.keys())
# n_boxes = len(d['text'])
# for i in range(n_boxes):
#     if int(d['conf'][i]) > 60:
#         (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#         img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.imshow('img', img)
# cv2.waitKey(0)


# import re
# import cv2
# import pytesseract
# from pytesseract import Output
# img = cv2.imread('FB_IMG_1579511433047.jpg')
# d = pytesseract.image_to_data(img, output_type=Output.DICT)
# keys = list(d.keys())
# date_pattern = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$'
# n_boxes = len(d['text'])
# for i in range(n_boxes):
#     if int(d['conf'][i]) > 60:
#     	if re.match(date_pattern, d['text'][i]):
# 	        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
# 	        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.imshow('img', img)
# cv2.waitKey(0)

# working code
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

# pdf_path = 'Modification-Type_2A.pdf'
# extracted_text = extract_text_with_ocr(pdf_path)
# print(extracted_text)
# print("-------------------------------------------------------------------")
# print("-------------------------------------------------------------------")
# print("-------------------------------------------------------------------")
# print("-------------------------------------------------------------------")

# # import cv2
# import fitz  # PyMuPDF
# import pytesseract
# import numpy as np

# # Preprocess the document
# def preprocess_document(document_path):
#     doc = fitz.open(document_path)
#     preprocessed_pages = []
#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)
#         pix = page.get_pixmap()
#         img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
#         if pix.n >= 4:  # RGBA
#             img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
#         img = fix_rotation(img)
#         img = adjust_zoom(img)
#         img = enhance_quality(img)
#         preprocessed_pages.append(img)
#     return preprocessed_pages

# def fix_rotation(image):
#     # Implement rotation correction logic
#     return image

# def adjust_zoom(image):
#     # Implement zoom adjustment logic
#     return image

# def enhance_quality(image):
#     # Implement quality enhancement logic
#     return image

# # Extract filing information from the first page
# def extract_filing_info(first_page_image):
#     ocr_result = pytesseract.image_to_string(first_page_image)
#     filing_info = parse_ocr_result(ocr_result)
#     return filing_info

# def parse_ocr_result(ocr_result):
#     # Implement OCR result parsing logic
#     filing_info = {
#         'tick_boxes': [
#             {'section_number': 1, 'checked': True},
#             {'section_number': 2, 'checked': False}
#         ]
#     }
#     return filing_info

# # Identify sections to process based on filing information
# def identify_sections(filing_info):
#     sections = []
#     for tick_box in filing_info['tick_boxes']:
#         if tick_box['checked']:
#             sections.append(tick_box['section_number'])
#     return sections

# # Extract section content based on section number
# def extract_section(preprocessed_pages, section_number):
#     # Implement logic to locate and extract section content
#     section_content = preprocessed_pages[section_number - 1]  # Assuming section_number corresponds to page number
#     return section_content

# # Extract data points from section content using templates
# def extract_data_points(section_content, section_number):
#     template = get_template_for_section(section_number)
#     data_points = apply_template(section_content, template)
#     return data_points

# def get_template_for_section(section_number):
#     # Implement logic to retrieve the template for the given section number
#     template = {
#         1: {"field1": "Label1", "field2": "Label2"},
#         2: {"field3": "Label3", "field4": "Label4"}
#     }
#     return template.get(section_number, {})

# def apply_template(section_content, template):
#     # Implement logic to apply template and extract data points
#     data_points = {}
#     ocr_result = pytesseract.image_to_string(section_content)
#     for field, label in template.items():
#         if label in ocr_result:
#             data_points[field] = ocr_result.split(label)[1].split('\n')[0].strip()
#     return data_points

# # Main function to process the document
# def process_document(document_path):
#     preprocessed_pages = preprocess_document(document_path)
#     filing_info = extract_filing_info(preprocessed_pages[0])
#     sections_to_process = identify_sections(filing_info)
#     results = {}

#     for section in sections_to_process:
#         section_content = extract_section(preprocessed_pages, section)
#         results[section] = extract_data_points(section_content, section)
    
#     return results

# # Example usage
# document_path = "/mnt/data/sample_document.pdf"
# results = process_document(document_path)
# print(results)


# # Example usage
# document_path = "Modification-Type_2A.pdf"
# results = process_document(document_path)
# print(results)
