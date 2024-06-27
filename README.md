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
uvicorn data_extraction_engine.app.main:app --reload# data_extraction_engine
