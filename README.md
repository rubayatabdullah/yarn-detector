# Weft Yarn Analyzer

A mobile-first Flask app for yarn acceptability checks in a lab environment. It stores reference yarn samples, measures yarn thickness from the image, and compares a new yarn sample against the saved reference using a ±15% tolerance window.

## Features

- Use the phone camera or upload a photo from the gallery.
- Save a reference yarn sample with its measured width.
- Compare a new yarn image against the saved reference.
- Return only an acceptability result: ACCEPTABLE or NOT ACCEPTABLE.
- Keep the result inline on the live camera page.
- Store yarn references in SQLite and saved photos in `data/training`.

## Local setup

1. Open a terminal in this folder.
2. Create and activate a virtual environment.
3. Install dependencies:

   `python -m pip install -r requirements.txt`

4. Create a `.env` file:

   `SECRET_KEY=your-secret-key`

5. Run the app:

   `python app.py`

6. Open http://localhost:5000 in your browser.

## Production deployment

This project is ready for Render or a similar Python host.

Recommended settings:

- start command: `gunicorn wsgi:app --bind 0.0.0.0:$PORT`
- environment variables: `SECRET_KEY` and `PORT`
- Python runtime: 3.12+

## Lab-use tips

- Keep the yarn straight and fully visible in frame.
- Use a fixed phone distance for every sample.
- Hold the camera steady and avoid glare.
- Use the same lighting conditions for both reference and test yarns.
