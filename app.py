from flask import Flask, render_template, request
import pandas as pd
import urllib.parse
from urllib.parse import unquote
from dotenv import load_dotenv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from bs4 import BeautifulSoup
import time
from selenium.webdriver.common.by import By
from collections import Counter
import re
import io
import base64
import arabic_reshaper
from bidi.algorithm import get_display
from wordcloud import WordCloud
import csv
import nltk
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import google.generativeai as genai
from safetensors.torch import load_file
import requests

# Load environment variables from .env file
load_dotenv()

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

from package import dynamic_scraper
from package import scrape_comments
# Import debug functions
import sys
sys.path.append('.')

app = Flask(__name__)

# Load stop words from the CSV file
stop_words_df = pd.read_csv('Stop_words.csv', header=None, encoding='utf-16')
stop_words = set(stop_words_df[0].tolist())  # Convert to a set for faster lookups


# Load the pre-trained tokenizer
tokenizer = BertTokenizer.from_pretrained("SI2M-Lab/DarijaBERT")

# Load the model structure from the same pre-trained model
model = BertForSequenceClassification.from_pretrained("SI2M-Lab/DarijaBERT", num_labels=2)

# Load the weights from the .safetensors file
safetensors_path = r"C:\Users\pc\Desktop\sentiment analysis\Darija-sentiment-Analysis\models\model\model.safetensors"
state_dict = load_file(safetensors_path)
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

def analyze_sentiment(comments):
    results = []
    all_words = []

    for comment_data in comments:
        comment = comment_data['comment']
        like_count = comment_data['likes']

        # Tokenize the comment with explicit max_length
        inputs = tokenizer(comment, return_tensors="pt", truncation=True, max_length=128, padding=True)

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).item()

            # Determine sentiment
            sentiment = 'positive' if predictions ==1 else 'negative'
            #print(f"Predicted Sentiment: {sentiment}")
            #print(f"prediction_value: {predictions}")
        # Append result
        results.append({
            'comment': comment,
            'likes': like_count,
            'sentiment': sentiment
        })

        # Tokenize and filter words (remove stopwords)
        words = re.findall(r'\b\w+\b', comment.lower())
        filtered_words = [word for word in words if word not in stop_words]
        all_words.extend(filtered_words)

    # Top 5 common words
    word_counts = Counter(all_words)
    top_5_words = word_counts.most_common(5)

    return results, top_5_words


# Function to load stopwords from stopwords.csv
def load_stopwords(file_path):
    all_stopwords = set()
    with open(file_path, 'r', encoding='utf-16') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:  # Check if row is not empty
                all_stopwords.add(row[0].strip())
    
    # Import and get Arabic stopwords from NLTK
    from nltk.corpus import stopwords
    try:
        arabic_stopwords = set(stopwords.words('arabic'))
        # Union arabic with darija stopwords
        all_stopwords = all_stopwords.union(arabic_stopwords)
    except LookupError:
        # If Arabic stopwords are not available, just use Darija ones
        print("Arabic stopwords not found, using only Darija stopwords")
    
    return all_stopwords


# Configure the Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Function to generate a sentiment report via Gemini-like API
def generate_report(positive_percentage, negative_percentage):
    # Create the prompt for the API using the positive and negative percentages
    prompt = f"The article has {positive_percentage}% positive comments and {negative_percentage}% negative comments. Based on this, generate a sentiment report in arabic summarizing the sentiments presenting  with the articles and by negative comment it means that reader is sad about the article."

    # Use the generative AI model to create a response
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    
    try:
        response = model.generate_content(prompt)
        # Extract and return the generated report
        return response.text
    
    except Exception as e:
        print(f"Error generating report: {e}")
        return "Could not generate the report due to an API error."

# Function to clean and tokenize the text, removing stopwords
def clean_and_tokenize_comments(comments, stop_words):
    all_words = []
    for comment_data in comments:
        comment = comment_data['comment']
        # Tokenize and filter words (remove stopwords)
        words = re.findall(r'\b\w+\b', comment.lower())
        filtered_words = [word for word in words if word not in stop_words]
        all_words.extend(filtered_words)
    return all_words

# Function to generate and display the word cloud
def generate_wordcloud(filtered_words):
    # Create a word cloud
    reshaped_text = arabic_reshaper.reshape(' '.join(filtered_words))
    bidi_text =get_display(reshaped_text)
    wordcloud = WordCloud(font_path='arial', 
                          background_color='white',
                          width=300,
                          height=350).generate(bidi_text)
    # Save wordcloud image to a BytesIO object
    img = io.BytesIO()
    #plot the wordcould image
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Convert image to base64
    wordcloud_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return wordcloud_url


# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to display results after scraping
@app.route('/results', methods=['POST'])
def results():
    key_name = request.form['keyword']
    interval_days = int(request.form['days'])
    url = 'https://www.hespress.com/all?most_commented'  # URL to scrape from
    
    df = dynamic_scraper(interval_days, key_name, url)
    print(df)
    if df is not None:
        # Show the scraped data on the results page
        return render_template('results.html', data=df.to_dict(orient='records'))
    else:
        return render_template('results.html', data=[])


# Route to display all comments and generate the pie chart
@app.route('/all_comments', methods=['POST'])
def all_comments():
    article_links = request.form.getlist('article_links')  # Get all article links from the form

    all_comments = []
    positive_count = 0
    negative_count = 0
    stop_words2 = load_stopwords('Stop_words.csv')
    # Scrape and analyze comments from each article
    for link in article_links:
        comments = scrape_comments(link)
        analyzed_comments, _ = analyze_sentiment(comments)
        all_comments.extend(analyzed_comments)

        # Count positive and negative comments
        for comment in analyzed_comments:
            if comment['sentiment'] == 'positive':
                positive_count += 1
            elif comment['sentiment'] == 'negative':
                negative_count += 1

    # Total number of comments
    total_comments = positive_count + negative_count

    # Calculate percentages (ensure no division by zero)
    if total_comments > 0:
        positive_percentage = (positive_count / total_comments) * 100
        negative_percentage = (negative_count / total_comments) * 100
    else:
        positive_percentage = 0
        negative_percentage = 0

    # Debugging: Print to check calculations
    print(f"Positive Count: {positive_count}, Negative Count: {negative_count}")
    print(f"Positive Percentage: {positive_percentage}%, Negative Percentage: {negative_percentage}%")

    # Generate multiple visualizations
    charts = {}
    
    if total_comments > 0:
        # 1. Sentiment Distribution Pie Chart
        labels = ['Positive', 'Negative']
        sizes = [positive_percentage, negative_percentage]
        colors = ['#4CAF50', '#f44336']
        
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors,
                explode=(0.05, 0.05), shadow=True)
        plt.title('Sentiment Distribution', fontsize=16, fontweight='bold')
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
        img.seek(0)
        charts['pie_chart'] = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()
        
        # 2. Sentiment Comparison Bar Chart
        plt.figure(figsize=(10, 6))
        sentiments = ['Positive', 'Negative']
        counts = [positive_count, negative_count]
        bars = plt.bar(sentiments, counts, color=['#4CAF50', '#f44336'], alpha=0.8)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.title('Sentiment Analysis Results', fontsize=16, fontweight='bold')
        plt.xlabel('Sentiment', fontsize=12)
        plt.ylabel('Number of Comments', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
        img.seek(0)
        charts['bar_chart'] = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()
        
        # 3. Engagement Analysis (if likes data available)
        positive_likes = sum(int(comment.get('likes', 0)) for comment in all_comments 
                           if comment['sentiment'] == 'positive')
        negative_likes = sum(int(comment.get('likes', 0)) for comment in all_comments 
                           if comment['sentiment'] == 'negative')
        
        if positive_likes > 0 or negative_likes > 0:
            plt.figure(figsize=(10, 6))
            engagement = ['Positive Comments', 'Negative Comments']
            likes = [positive_likes, negative_likes]
            bars = plt.bar(engagement, likes, color=['#4CAF50', '#f44336'], alpha=0.8)
            
            for bar, like_count in zip(bars, likes):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(likes) * 0.01,
                        str(like_count), ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            plt.title('Engagement by Sentiment (Total Likes)', fontsize=16, fontweight='bold')
            plt.xlabel('Sentiment Category', fontsize=12)
            plt.ylabel('Total Likes', fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
            img.seek(0)
            charts['engagement_chart'] = base64.b64encode(img.getvalue()).decode('utf8')
            plt.close()
        
        # 4. Comment Length Analysis
        comment_lengths = [len(comment['comment'].split()) for comment in all_comments]
        positive_lengths = [len(comment['comment'].split()) for comment in all_comments 
                          if comment['sentiment'] == 'positive']
        negative_lengths = [len(comment['comment'].split()) for comment in all_comments 
                          if comment['sentiment'] == 'negative']
        
        plt.figure(figsize=(12, 6))
        plt.hist([positive_lengths, negative_lengths], bins=20, alpha=0.7, 
                label=['Positive Comments', 'Negative Comments'], 
                color=['#4CAF50', '#f44336'])
        plt.title('Comment Length Distribution by Sentiment', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Words', fontsize=12)
        plt.ylabel('Number of Comments', fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
        img.seek(0)
        charts['length_chart'] = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()
    
    # Calculate additional statistics
    stats = {
        'total_articles': len(article_links),
        'total_comments': total_comments,
        'avg_comments_per_article': round(total_comments / len(article_links), 2) if article_links else 0,
        'avg_comment_length': round(sum(len(comment['comment'].split()) for comment in all_comments) / total_comments, 2) if total_comments > 0 else 0,
        'most_positive_article': None,
        'most_negative_article': None,
        'engagement_rate': round((positive_likes + negative_likes) / total_comments, 2) if total_comments > 0 and 'positive_likes' in locals() else 0
    }
    # Generate AI sentiment report
    sentiment_report = generate_report(positive_percentage, negative_percentage)
    
    # Generate the word cloud from all comments
    filtered_words = clean_and_tokenize_comments(all_comments, stop_words2)
    wordcloud_url = generate_wordcloud(filtered_words)

    # Pass comprehensive data to the template
    return render_template(
        'all_comments.html',
        sentiment_results=all_comments,
        positive_count=positive_count,
        negative_count=negative_count,
        positive_percentage=positive_percentage,
        negative_percentage=negative_percentage,
        charts=charts,
        stats=stats,
        sentiment_report=sentiment_report,
        wordcloud_url=wordcloud_url,
        total_comments=total_comments
    )



@app.route('/analyze_comments/<path:article_link>', methods=['GET'])
def analyze_comments(article_link):
    article_link = unquote(article_link)  # Decode the URL-encoded article link
    print("Decoded Article Link:", article_link) 
    comments = scrape_comments(article_link)  # Scrape comments
    # Debugging: Check if comments are scraped
    print("Scraped Comments:", comments)
    
    if not comments:
        print("No comments found")
    
    sentiment_results, top_5_words = analyze_sentiment(comments)  # Analyze sentiment and get top 5 redundant words
    
    # Print the top 5 redundant words for debugging
    print("Top 5 Redundant Words:", top_5_words)
    
    return render_template('comments.html', sentiment_results=sentiment_results, top_5_words=top_5_words)



if __name__ == '__main__':
    app.run(debug=True)