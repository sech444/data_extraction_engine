Setup and Configuration Using Python Poetry

    Initialize Python Environment with Poetry
        Create a new project using Poetry.
        # Add Poetry to PATH
        export PATH="$HOME/.local/bin:$PATH"

        Install necessary libraries using Poetry.

    Define Project Structure
        Organize the project into meaningful directories and files.

    Install Necessary Libraries
        Use Poetry to manage dependencies such as PyMuPDF, OpenCV, Tesseract, SQLite, and FastAPI.

    Database Initialization
        Initialize an SQLite database for storing extracted data points.

    Implement PDF Processing Functions
        Write functions for PDF processing, image preprocessing, anchor detection, DP extraction, and checkbox detection.

    Define FastAPI Endpoints
        Define API endpoints for submitting PDF files and querying extracted data.

# Activate the poetry environment and run the FastAPI server
poetry shell
uvicorn data_extraction_engine.app.main:app --reload # data_extraction_engine
# data_extraction_engine

User Avatar
 Some tips for the project:

If you look into any of the documents, page 1 will contain a table that indicates what type of filings and subsequently datapoints (DP’s) you can expect in the document.

So it makes sense to develop a modular approach per DP group per Section.

Each section is marked with the number on the left. Position of the section may change, but DP position RELATIVE TO SECTION number and some anchors will always be the same.

So we can then approach the situation as follows:

1/ fix rotation, zoom, enhance quality
2/ identify what kind of filings do we expect in the document (page 1 and tick boxes)
3/ locate section based on tick boxes
4/ apply template / read section to extract DP's
To speed things up, we can also consider using “masks” that read only sections where DP’s are expected, based on the section number. This will increase the reading time significantly.
Sending sample for reference.
But at the end you are free to choose your approach 
