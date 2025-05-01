 YouTube Comment Sentiment Analyzer

A Streamlit application that analyzes the sentiment of YouTube video comments using machine learning. The app fetches comments from YouTube videos and performs sentiment analysis to classify them as positive or negative.

## Features

- Fetch comments from any YouTube video using the video ID
- Analyze sentiment of comments using a fine-tuned BERT model
- Interactive visualization of sentiment distribution
- Configurable number of comments to analyze
- Clean and intuitive user interface

## Requirements

The application requires the following Python packages:

```
streamlit==1.24.0
transformers==4.30.2
google-api-python-client==2.70.0
pandas==1.5.3
plotly==5.11.0
torch>=2.0.0
typing-extensions>=4.5.0
```

## Installation

1. Clone this repository or download the source code
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

1. Get a YouTube Data API Key:
   - Go to the [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Enable the YouTube Data API v3
   - Create credentials (API key)
   - Copy your API key

## Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. In the application:
   - Enter your YouTube Data API key in the sidebar
   - Paste a YouTube video ID or URL
   - Adjust the maximum number of comments to analyze using the slider
   - Click "Analyze" to start the analysis

3. View the results:
   - See the sentiment distribution chart
   - Browse the detailed comment analysis in the table

## Notes

- The maximum number of comments that can be fetched is limited to 50 per request
- Comments are cleaned before analysis to remove URLs, mentions, and special characters
- The sentiment analysis model is specifically fine-tuned for YouTube comments

## Error Handling

The application includes error handling for:
- Missing API key or video ID
- Invalid API key or video ID
- Failed comment fetching
- Sentiment analysis errors

## Dependencies

- Streamlit for the web interface
- Transformers (BERT) for sentiment analysis
- YouTube Data API for fetching comments
- Pandas for data manipulation
- Plotly for visualization

## Deployment on Streamlit Cloud

1. Create a Streamlit account at [share.streamlit.io](https://share.streamlit.io)

2. Connect your GitHub repository:
   - Fork or push this repository to your GitHub account
   - Log in to Streamlit Cloud
   - Click on "New app"
   - Select your repository and branch
   - Set the main file path as `app.py`

3. Configure environment variables:
   - In the Streamlit Cloud dashboard, go to your app's settings
   - Add your YouTube API key as a secret:
     - Name: `YOUTUBE_API_KEY`
     - Value: Your API key

4. Deploy:
   - Click "Deploy"
   - Wait for the build process to complete
   - Your app will be live at a unique URL

Note: Make sure your `app.py` reads the API key from environment variables:
```python
import os
api_key = os.getenv('YOUTUBE_API_KEY')
```
