#%% Import libraries
import seaborn as sns
import pandas as pd
from googleapiclient.discovery import build
import numpy as np
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
from textblob import TextBlob
from datetime import datetime
from tabulate import tabulate
import time
import requests

#%% Declare variables
api_key='Your API key'
youtube = build('youtube','v3',developerKey=api_key)

#%% Define a function to search in youtube with a given country code
def search_channel(search_term):
    request = youtube.search().list(
        part='snippet',
        q=search_term,
        type='channel',
        maxResults=50)
    response = request.execute()
    return response['items']
#%% Search terms
search_term=input('Enter the topic to search YouTube channels:') 
channel_data=[]
search_results=search_channel(search_term)
for item in search_results:
    channel_id=item['snippet']['channelId']
    channel_data.append({'id': channel_id})
#%% Function to get channel details using channel ID
def get_channel_stats(channel_id):
    request=youtube.channels().list(
        part='snippet,statistics',
        id=channel_id)
    response=request.execute()
    return response['items'][0]
#%% Get subscriber counts for each channel
for channel in channel_data:
    channel_stats=get_channel_stats(channel['id'])
    subscriber_count=channel_stats['statistics'].get('subscriberCount',0)
    channel['subscribers']=int(subscriber_count)
    time.sleep(3)

#%% Print all channel stats
i=1
for channel in channel_data:
    channel_stats=get_channel_stats(channel['id'])
    print(i,'Channel title: ',channel_stats['snippet']['title'])
    statistics=channel_stats['statistics']
    print('   Subscribers: {:,}'.format(int(statistics.get('subscriberCount',0))))
    print('   Video count: {:,}'.format(int(statistics.get('videoCount',0))))
    print('   View count: {:,}'.format(int(statistics.get('viewCount',0))))
    i=i+1
    time.sleep(3)

#%% Selected channel(Supply Chain Secrets)
selected_channel_id = 'Your selected channel ID'  

#%% Function for channel statistics
def get_channel_stats(youtube, selected_channel_id):
    request = youtube.channels().list(
        part='snippet,contentDetails,statistics',
        id=selected_channel_id
    )
    response = request.execute()

    data = {
        'Channel_name': response['items'][0]['snippet']['title'],
        'Subscribers': response['items'][0]['statistics']['subscriberCount'],
        'Views': response['items'][0]['statistics']['viewCount'],
        'Total_videos': response['items'][0]['statistics']['videoCount'],
        'playlist_id': response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    }
    return data
#%% Selectecd channel stats
selected_channel_stats=get_channel_stats(youtube,selected_channel_id)  

#%% Convert statistics into dataframe 
selected_channel_data = pd.DataFrame([selected_channel_stats])
print(selected_channel_data)


#%% Get the playlist_id for the selected channel
playlist_id = selected_channel_data.loc[
    selected_channel_data['Channel_name'].str.strip() == 'Selected Channel Name', 
    'playlist_id'
].iloc[0]
#%%Function for getting video Ids
def get_video_ids(youtube, playlist_id):
    request = youtube.playlistItems().list(
              part='contentDetails', playlistId = playlist_id,
              maxResults=200)
    response = request.execute()
    
    video_ids=[]
    for i in range(len(response['items'])):
        video_ids.append(response['items'][i]['contentDetails']['videoId'])
    
    return video_ids
#%% Retrieve video ids into a variable
video_ids=get_video_ids(youtube, playlist_id)
#%% Function to retrieve video details 
def get_video_details(youtube, video_ids):
    all_video_stats = []
    request = youtube.videos().list(
        part='snippet,statistics,contentDetails',
        id=','.join(video_ids)
    )

    response = request.execute()
    time.sleep(1)

    for video in response['items']:
        video_stats = {
            # Video Metadata
            'Title': video['snippet']['title'],
            'Description': video['snippet'].get('description', ''),
            'Tags': video['snippet'].get('tags', []),
            'CategoryId': video['snippet'].get('categoryId', ''),
            
            # Publishing Information
            'Published_date': video['snippet']['publishedAt'],
            'Duration': video['contentDetails'].get('duration', ''),
            
            # Engagement Metrics
            'Views': video['statistics'].get('viewCount', 0),
            'Likes': video['statistics'].get('likeCount', 0),
            'Comments': video['statistics'].get('commentCount', 0),
            
            # Content Elements
            'Thumbnail': video['snippet']['thumbnails']['high']['url'],
            'Video_link': f"https://www.youtube.com/watch?v={video['id']}"
        }

        all_video_stats.append(video_stats)

    return all_video_stats
#%% Retrieve video details
video_details = get_video_details(youtube, video_ids)
video_data = pd.DataFrame(video_details)

#%% Data preprocessing
# Handling missing values
video_data.fillna({
    'Description': '',
    'Tags': '[]',
    'Views': 0,
    'Likes': 0,
    'Comments': 0
}, inplace=True)

# Standardizing date formats
video_data['Published_date'] = pd.to_datetime(video_data['Published_date'])

# Normalizing text data (titles, descriptions)
video_data['Title'] = video_data['Title'].str.strip().str.lower()
video_data['Description'] = video_data['Description'].str.strip().str.lower()

# Converting metrics to appropriate data types
video_data['Views'] = pd.to_numeric(video_data['Views'])
video_data['Likes'] = pd.to_numeric(video_data['Likes'])
video_data['Comments'] = pd.to_numeric(video_data['Comments'])

# Removing duplicate entries
video_data.drop_duplicates(subset='Title', inplace=True)

# Engagement rate = (Likes + Comments) / Views * 100
video_data['Engagement_rate'] = ((video_data['Likes'] + video_data['Comments']) / video_data['Views']) * 100

#%% Data Analysis

#Descriotive Statistics
desc_stats = video_data[['Views', 'Likes', 'Comments', 'Engagement_rate']].describe().T

# Rename columns for clarity
desc_stats.rename(columns={
    'count': 'Count',
    'mean': 'Mean',
    'std': 'Std Dev',
    'min': 'Min',
    '25%': '25th Percentile',
    '50%': 'Median',
    '75%': '75th Percentile',
    'max': 'Max'
}, inplace=True)

# Round the values for neat display
desc_stats = desc_stats.round(2)

# Display the table
print(tabulate(desc_stats, headers='keys', tablefmt='grid'))


# Time series analysis of publishing patterns
plt.figure(figsize=(10,6))
video_data['Published_date'].dt.to_period('M').value_counts().sort_index().plot(kind='line')
plt.title('Publishing Pattern Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Videos')
plt.show()

# Content analysis: keyword frequency (from titles)
all_words = ' '.join(video_data['Title'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Video Titles')
plt.show()

# Title length distribution
video_data['Title_length'] = video_data['Title'].apply(len)
plt.figure(figsize=(10,5))
sns.histplot(video_data['Title_length'], bins=20)
plt.title('Title Length Distribution')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.show()

# Engagement analysis: correlation between video characteristics and engagement
corr = video_data[['Views', 'Likes', 'Comments', 'Engagement_rate', 'Title_length']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

#%% Sentiment Analysis
def get_video_comments(youtube, video_id, max_comments=200):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=200,  # Max 100 per page
            textFormat="plainText"
        )
        response = request.execute()
        
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            
            # Stop if we've hit the max_comments
            if len(comments) >= max_comments:
                break

    except Exception as e:
        print(f"Error fetching comments for video {video_id}: {e}")
    
    return comments

all_comments = []

for vid in video_ids:
    video_comments = get_video_comments(youtube, vid, max_comments=20)
    all_comments.extend(video_comments)
    print(f"Fetched {len(video_comments)} comments from video {vid}")

print(f"Total comments fetched: {len(all_comments)}")

#%%
from textblob import TextBlob

# Convert to DataFrame
comments_df = pd.DataFrame({'Comment': all_comments})

# Apply sentiment analysis
comments_df['Polarity'] = comments_df['Comment'].apply(lambda x: TextBlob(x).sentiment.polarity)
comments_df['Subjectivity'] = comments_df['Comment'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Compute average polarity and subjectivity
avg_polarity = comments_df['Polarity'].mean()
avg_subjectivity = comments_df['Subjectivity'].mean()

# Determine overall sentiment label
if avg_polarity > 0:
    sentiment_label = 'positive'
elif avg_polarity < 0:
    sentiment_label = 'negative'
else:
    sentiment_label = 'neutral'

print(f"Sentiment Label: {sentiment_label}")
print(f"Average Polarity: {avg_polarity:.3f}")
print(f"Average Subjectivity: {avg_subjectivity:.3f}")

#%% Bar chart of average polarity and subjectivity
plt.figure(figsize=(8,6))
plt.bar(['Polarity', 'Subjectivity'], [avg_polarity, avg_subjectivity], color=['green', 'blue'])
plt.title('Average Sentiment Scores from YouTube Comments')
plt.ylabel('Score')
plt.ylim(-1, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#%% WordCloud from all individual comments
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(' '.join(comments_df['Comment']))

# Display WordCloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of YouTube Comments")
plt.show()

#%% Video Performance Segmentation based on View Count

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Convert Views column to numeric if not already
video_data['Views'] = pd.to_numeric(video_data['Views'], errors='coerce')

# Define quantile thresholds
low_thresh = video_data['Views'].quantile(0.33)
high_thresh = video_data['Views'].quantile(0.66)

# 2Segment videos
def categorize_performance(views):
    if views < low_thresh:
        return 'Low'
    elif views < high_thresh:
        return 'Medium'
    else:
        return 'High'

video_data['Performance_Segment'] = video_data['Views'].apply(categorize_performance)

# Display sample with new column
print(video_data[['Title', 'Views', 'Performance_Segment']].head())

#%% Visualize the segmentation
plt.figure(figsize=(8,5))
sns.countplot(data=video_data, x='Performance_Segment', palette='Set2')
plt.title('Video Performance Segmentation (Based on View Count)')
plt.xlabel('Performance Segment')
plt.ylabel('Number of Videos')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
