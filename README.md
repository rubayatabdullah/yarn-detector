# Yarn Color Lab

A small mobile-friendly Flask app that learns yarn colors from photos you label and then matches new photos to your saved palette.

## Features

- Upload images from a phone camera using the mobile capture input.
- Save a photo with a color label such as Red, Olive, or Navy.
- Learn the dominant color from each saved yarn sample.
- Analyze new yarn photos with Gemini and compare the result against your saved colors.
- Store your learned palette locally in the data folder.

## Setup

1. Open a terminal in this folder.
2. Install dependencies:

   `python -m pip install -r requirements.txt`

3. Create a `.env` file from the example:

   `copy .env.example .env`

4. Add your Gemini API key:

   `GEMINI_API_KEY=your_key_here`

5. Run the app:

   `python app.py`

6. Open http://localhost:5000 in your browser.

## Notes

- The app saves learned samples in `data/learned_palette.json` and stores the image files under `data/training`.
- The web upload input uses `capture="environment"` so it works well with phone cameras.
- If the Gemini API key is missing, the app still offers the local color-matching logic with a warning message.
