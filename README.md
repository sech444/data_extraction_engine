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

# this is the pdf format we are working with

NOTE: In 2A/2B forms,
number after dot
indicates number of
managers filing is made
for. so 11.1 is details of
manager #1, 11.2 would
be for manager #2 etc

in the new form, there is a
box to determine what
kind of manager this is.
physical person
(Personne physique) or
legal entity (Perso



NOTE: In 2A/2B forms,
number after dot
indicates number of
managers filing is made
for. so 11.1 is details of
manager #1, 11.2 would
be for manager #2 etc.
DP_049
DP_010
NOTE: Managers/Directors can be either
private individuals (personne physique) or
legal entities (person morale). In case of S.A.
or S.E. (legal types of the entities), on top of
management, there can be one permanent
representative (same director from the
Board). Hence why the section is so long
FYI
Board: 2+ Managers/Directors
Director: S.A. or S.E.
Manager: all other legal entity types
DP_011
in the new form, there is a
box to determine what
kind of manager this is.
physical person
(Personne physique) or
legal entity (Personne
morale)
DP_012
DP_013
DP_014
DP_015
DP_016
DP_017
DP_018
DP_019
DP_020
DP_021
DP_022
DP_023
DP_024
DP_025
DP_026
DP_027

DP_049

Type something

NOTE: the way this section works is as
follows. Manager/Director is appointed either
till the end of time, or until defined date. Date
can be either specific (like 01/01/2028) OR
the date of next AGM to be held (in this case
2014). When recording data in our DB, we
have to take this into consideration