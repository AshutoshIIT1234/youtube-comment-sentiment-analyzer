import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import re
from typing import List, Dict
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import plotly.express as px
import asyncio  # Import asyncio
import sys
from urllib.parse import urlparse, parse_qs

# Initialize the Sentiment Analysis Pipeline
try:
    # Load the specific model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("rahulk98/bert-finetuned-youtube_sentiment_analysis")
    model = AutoModelForSequenceClassification.from_pretrained("rahulk98/bert-finetuned-youtube_sentiment_analysis")
    # Explicitly move model to CPU and configure pipeline
    model = model.to('cpu')
    sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, device='cpu')
except Exception as e:
    st.error(f"Error initializing sentiment analysis pipeline: {e}. Please ensure you have the necessary libraries installed and the model is available.")
    st.error(
        "Please install the required libraries by running the following command in your terminal:\n"
        "`pip install streamlit transformers google-api-python-client pandas plotly`"
    )
    # You might want to halt execution here if the pipeline is essential
    sys.exit(1)  # Stop if the pipeline fails to initialize


# Function to Clean Text
def clean_text(text: str) -> str:
    text = re.sub(r"http\S|www\S|https\S", "", text, flags=re.IGNORECASE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.strip()
    return text

# Function to Analyze Sentiment of a Single Comment
def analyze_comment_sentiment(comment: str) -> Dict:
    cleaned_comment = clean_text(comment)
    if not cleaned_comment:
        return {}
    try:
        result = sentiment_pipeline(cleaned_comment)[0]
        return result
    except Exception as e:
        st.error(f"Error analyzing comment: {e}.  Returning empty result for this comment.")
        return {}

# Function to Analyze Sentiment of Multiple Comments
def analyze_comments_sentiment(comments: List[str]) -> List[Dict]:
    results = []
    for comment in comments:
        results.append(analyze_comment_sentiment(comment))
    return results

# Function to Get YouTube Comments
def get_youtube_comments(video_id: str, api_key: str, max_results: int = 100) -> List[str]:
    """
    Retrieves comments from a YouTube video using the YouTube Data API.

    Args:
        video_id (str): The ID of the YouTube video.
        api_key (str): Your YouTube Data API key.
        max_results (int, optional): The maximum number of comments to retrieve. Defaults to 100.
            Maximum is 50, and if the user provides a number greater than 50, it will be set to 50.

    Returns:
        List[str]: A list of comment texts.  Returns an empty list if there are errors or no comments.
    """
    comments = []
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        if max_results > 50:
            max_results = 50

        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText"
        )
        response = request.execute()

        for item in response.get("items", []):
            comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment_text)

    except HttpError as e:
        if e.resp.status == 404:
            st.error(f"Error: The video ID '{video_id}' could not be found. Please check the ID and try again.")
            return []
        else:
            st.error(f"YouTube API Error: {e}")
            return []  # Return an empty list in case of an error
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return []

    return comments

def create_sentiment_dataframe(comments: List[str], sentiment_results: List[Dict]) -> pd.DataFrame:
    """
    Creates a Pandas DataFrame from the comments and their sentiment analysis results.

    Args:
        comments (List[str]): A list of comments.
        sentiment_results (List[Dict]): A list of sentiment analysis results
            (dictionaries) corresponding to the comments.

    Returns:
        pd.DataFrame: A DataFrame with columns 'Comment', 'Sentiment', and 'Score'.
                    Returns an empty DataFrame if there are no comments.
    """
    if not comments:
        return pd.DataFrame()

    # Create lists for the DataFrame, handling empty results
    sentiments = []
    scores = []
    for result in sentiment_results:
        if result:
            sentiments.append(result['label'])
            scores.append(result['score'])
        else:
            sentiments.append('No Sentiment')  # consistent no sentiment label
            scores.append(0.0)  # Provide a default score

    df = pd.DataFrame({'Comment': comments, 'Sentiment': sentiments, 'Score': scores})
    return df


def generate_sentiment_visualization(df: pd.DataFrame): # -> px.bar.Figure:
    """
    Generates a Plotly bar chart visualizing the sentiment distribution from the DataFrame.

    Args:
        df (pd.DataFrame): A DataFrame with 'Sentiment' and 'Score' columns.

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly bar chart. # Changed return type annotation
    """
    if df.empty:
        fig = px.bar() # Return empty plot
        fig.update_layout(title="No Sentiments to Display")
        return fig

    sentiment_counts = df['Sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']  # Rename for clarity

    fig = px.bar(sentiment_counts, x='Sentiment', y='Count',
                 title='Sentiment Distribution of YouTube Comments',
                 color='Sentiment',
                 color_discrete_map={'POSITIVE': 'green',
                                     'NEGATIVE': 'red',
                                     'No Sentiment': 'grey'}) #Added a color for "No Sentiment"

    return fig



def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("YouTube Comment Sentiment Analyzer")

    # Sidebar for user input
    youtube_api_key = st.sidebar.text_input("Enter your YouTube Data API Key:", type="password")
    video_id_or_url = st.sidebar.text_input("Enter the YouTube Video ID or URL:")
    max_comments = st.sidebar.slider("Max Comments", min_value=1, max_value=500, value=100, step=10) #Added a slider

    if st.sidebar.button("Analyze"):
        if not youtube_api_key:
            st.error("Please enter your YouTube Data API key.")
            return
        if not video_id_or_url:
            st.error("Please enter a YouTube Video ID or URL.")
            return

        # Extract video ID if it's a URL
        if "youtube.com" in video_id_or_url:
            try:
                parsed_url = urlparse(video_id_or_url)
                query_params = parse_qs(parsed_url.query)
                video_id = query_params.get("v", [None])[0]  # Get the 'v' parameter
                if not video_id:
                    st.error("Invalid YouTube URL.  Could not extract video ID.")
                    return
            except Exception:
                st.error("Error parsing YouTube URL. Please enter a valid URL or Video ID.")
                return
        else:
            video_id = video_id_or_url # Assume it is a video ID

        st.info("Fetching comments from YouTube...")
        comments = get_youtube_comments(video_id, youtube_api_key, max_comments) # Pass max_comments
        if not comments:
            st.warning("No comments found for this video, or there was an error fetching them.")
            return

        st.info("Analyzing sentiment...")
        sentiment_results = analyze_comments_sentiment(comments)
        df = create_sentiment_dataframe(comments, sentiment_results)

        st.info("Generating visualization...")
        sentiment_chart = generate_sentiment_visualization(df)

        # Display results
        st.header("Sentiment Analysis Results")
        st.plotly_chart(sentiment_chart)

        st.dataframe(df)  # Show the DataFrame

#Wrap the main function call in the if __name__ == "__main__": block
if __name__ == "__main__":
    # Check if an event loop is already running
    if 'ipykernel' in sys.modules:
        # Code is running in a Jupyter Notebook, no need to create a new loop
        main()
    else:
        # Code is running in a normal Python environment
        try:
            asyncio.get_running_loop()
        except RuntimeError:  # 'There is no current event loop in thread'
            asyncio.set_event_loop(asyncio.new_event_loop())
        main()