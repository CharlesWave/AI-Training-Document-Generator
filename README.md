# AI Training Document Generator

## Problem Addressed:
Senior staff spend excessive time onboarding new hires; video recordings are inefficient for knowledge transfer

This application generates comprehensive training documents from one training session videos using the Google Gemini API. So that the document can be resued for faster onboarding new employees with minimum input from senior staff.


## Features

- 🎥 Upload training videos (MP4, AVI, MOV, WMV, MKV formats up to 500MB)
- 🤖 Process videos using Google Gemini API in a single stage
- 📷 Automatically extract screenshots at relevant timestamps
- 📄 Generate a training document with text and screenshots
- 📑 Structured document with summary, key points, and screenshots with captions

## How It Works

1. **Video Upload**: Upload your training video through the web interface. Note that currently the MVP solution can only take in video less than 20 minutes. 
2. **AI Analysis**: Gemini AI analyzes the video content to extract knowledge points and timestamps
3. **Screenshot Extraction**: The app extracts screenshots at the specified timestamps
4. **Document Generation**: A complete training document is created with knowledge points and screenshots

## Setup

### Prerequisites

- Python 3.8+
- Google Gemini API key
- OpenCV (for screenshot extraction)
- Flask (for the web application)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/training-document-generator.git
   cd training-document-generator
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Running the Application

1. Start the Flask application:
   ```
   python single_stage_app.py
   ```

2. Open your browser and go to:
   ```
    http://127.0.0.1:8080
   ```

3. Upload a training video and follow the on-screen instructions

## Project Structure

- `single_stage_app.py`: Main Flask application (It is called single stage because the current V1 only calls API once. Development is undergoing for improving output quality with multiple chained API call)
- `ai_service.py`: Handles API calls to Gemini
- `prompts_OneStage.py`: Contains the prompt used for Gemini API
- `templates/`: HTML templates for the web interface
- `uploads/`: Directory for storing uploaded videos
- `screenshots/`: Directory for storing extracted screenshots
- `output/`: Directory for storing generated documents

## API Documentation

### Gemini API

The application uses the Google Gemini API to analyze the video content. The prompt used for the API is defined in `prompts_OneStage.py`. The API is responsible for:

- Analyzing the video content
- Extracting knowledge points
- Identifying relevant timestamps for screenshots
- Generating captions for the screenshots

## Troubleshooting

### Common Issues

1. **Video Upload Fails**
   - Ensure the video format is supported (MP4, AVI, MOV, WMV, MKV)
   - Check that the file size is less than 500MB

2. **Gemini API Processing Fails**
   - Verify your API key is correct in the `.env` file
   - Check internet connectivity
   - Ensure the video is not corrupt

3. **Screenshot Extraction Issues**
   - Make sure OpenCV is properly installed
   - Check if the video can be opened by OpenCV

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini API for video content analysis
- OpenCV for video processing and screenshot extraction
- Flask for the web application framework 
