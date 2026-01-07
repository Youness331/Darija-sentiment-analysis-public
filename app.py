from flask import Flask, render_template, request, session
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
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

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
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-here")  # Add to .env file

# Global cache for storing scraped comments
# Structure: {article_link: {'comments': [...], 'timestamp': datetime}}
COMMENTS_CACHE = {}

# Load stop words from the CSV file
stop_words_df = pd.read_csv('Stop_words.csv', header=None, encoding='utf-16')
stop_words = set(stop_words_df[0].tolist())

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


def scrape_single_article_comments(link):
    """Helper function to scrape comments for a single article"""
    try:
        print(f"Scraping comments from: {link}")
        comments = scrape_comments(link)
        return link, comments
    except Exception as e:
        print(f"Error scraping {link}: {e}")
        return link, []


def scrape_all_comments_parallel(article_links, max_workers=3):
    """
    Scrape comments from multiple articles in parallel
    max_workers: number of concurrent scraping threads (adjust based on your system)
    """
    comments_by_article = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all scraping tasks
        future_to_link = {
            executor.submit(scrape_single_article_comments, link): link 
            for link in article_links
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_link):
            link, comments = future.result()
            comments_by_article[link] = comments
            print(f"Completed scraping {link}: {len(comments)} comments found")
    
    return comments_by_article


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
            sentiment = 'positive' if predictions == 1 else 'negative'

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


def load_stopwords(file_path):
    all_stopwords = set()
    with open(file_path, 'r', encoding='utf-16') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                all_stopwords.add(row[0].strip())
    
    from nltk.corpus import stopwords
    try:
        arabic_stopwords = set(stopwords.words('arabic'))
        all_stopwords = all_stopwords.union(arabic_stopwords)
    except LookupError:
        print("Arabic stopwords not found, using only Darija stopwords")
    
    return all_stopwords


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def generate_report(positive_percentage, negative_percentage):
    prompt = f"The article has {positive_percentage}% positive comments and {negative_percentage}% negative comments. Based on this, generate a sentiment report in arabic summarizing the sentiments presenting with the articles and by negative comment it means that reader is sad about the article."

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating report: {e}")
        return "Could not generate the report due to an API error."


def clean_and_tokenize_comments(comments, stop_words):
    all_words = []
    for comment_data in comments:
        comment = comment_data['comment']
        words = re.findall(r'\b\w+\b', comment.lower())
        filtered_words = [word for word in words if word not in stop_words]
        all_words.extend(filtered_words)
    return all_words


def generate_wordcloud(filtered_words):
    reshaped_text = arabic_reshaper.reshape(' '.join(filtered_words))
    bidi_text = get_display(reshaped_text)
    wordcloud = WordCloud(font_path='arial', 
                          background_color='white',
                          width=300,
                          height=350).generate(bidi_text)
    
    img = io.BytesIO()
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(img, format='png')
    img.seek(0)
    
    wordcloud_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return wordcloud_url


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/results', methods=['POST'])
def results():
    key_name = request.form['keyword']
    interval_days = int(request.form['days'])
    url = 'https://www.hespress.com/all?most_commented'
    
    # Step 1: Scrape articles
    print("Step 1: Scraping articles...")
    df = dynamic_scraper(interval_days, key_name, url)
    
    if df is not None and len(df) > 0:
        article_links = df['link'].tolist()
        
        # Step 2: Scrape comments from all articles in parallel
        print(f"Step 2: Scraping comments from {len(article_links)} articles in parallel...")
        comments_by_article = scrape_all_comments_parallel(article_links, max_workers=3)
        
        # Step 3: Store in cache and session
        global COMMENTS_CACHE
        COMMENTS_CACHE.update(comments_by_article)
        
        # Also store in session for persistence across requests
        session['comments_cache'] = {
            link: comments for link, comments in comments_by_article.items()
        }
        session['article_links'] = article_links
        
        # Add comment count to dataframe
        df['comment_count'] = df['link'].apply(lambda x: len(comments_by_article.get(x, [])))
        
        print(f"✓ Successfully scraped and cached comments for {len(article_links)} articles")
        
        return render_template('results.html', 
                             data=df.to_dict(orient='records'),
                             comments_cached=True)
    else:
        return render_template('results.html', data=[], comments_cached=False)


@app.route('/all_comments', methods=['POST'])
def all_comments():
    article_links = request.form.getlist('article_links')
    
    # Try to get cached comments first
    comments_cache = session.get('comments_cache', {})
    global COMMENTS_CACHE
    
    all_comments = []
    positive_count = 0
    negative_count = 0
    stop_words2 = load_stopwords('Stop_words.csv')
    
    # Use cached comments if available
    for link in article_links:
        # Check session cache first, then global cache
        comments = comments_cache.get(link) or COMMENTS_CACHE.get(link)
        
        if comments is None:
            # Fallback: scrape if not in cache
            print(f"Cache miss for {link}, scraping now...")
            comments = scrape_comments(link)
            COMMENTS_CACHE[link] = comments
        else:
            print(f"Cache hit for {link}: {len(comments)} comments")
        
        analyzed_comments, _ = analyze_sentiment(comments)
        all_comments.extend(analyzed_comments)

        for comment in analyzed_comments:
            if comment['sentiment'] == 'positive':
                positive_count += 1
            elif comment['sentiment'] == 'negative':
                negative_count += 1

    total_comments = positive_count + negative_count

    if total_comments > 0:
        positive_percentage = (positive_count / total_comments) * 100
        negative_percentage = (negative_count / total_comments) * 100
    else:
        positive_percentage = 0
        negative_percentage = 0

    print(f"Positive Count: {positive_count}, Negative Count: {negative_count}")
    print(f"Positive Percentage: {positive_percentage}%, Negative Percentage: {negative_percentage}%")

    # Generate visualizations
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
        
        # 3. Engagement Analysis
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
    
    # Calculate statistics
    stats = {
        'total_articles': len(article_links),
        'total_comments': total_comments,
        'avg_comments_per_article': round(total_comments / len(article_links), 2) if article_links else 0,
        'avg_comment_length': round(sum(len(comment['comment'].split()) for comment in all_comments) / total_comments, 2) if total_comments > 0 else 0,
        'engagement_rate': round((positive_likes + negative_likes) / total_comments, 2) if total_comments > 0 and 'positive_likes' in locals() else 0
    }
    
    # Generate AI sentiment report
    sentiment_report = generate_report(positive_percentage, negative_percentage)
    
    # Generate word cloud
    filtered_words = clean_and_tokenize_comments(all_comments, stop_words2)
    wordcloud_url = generate_wordcloud(filtered_words)

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
    article_link = unquote(article_link)
    print("Decoded Article Link:", article_link)
    
    # Try to get cached comments
    comments_cache = session.get('comments_cache', {})
    global COMMENTS_CACHE
    
    comments = comments_cache.get(article_link) or COMMENTS_CACHE.get(article_link)
    
    if comments is None:
        # Fallback: scrape if not in cache
        print(f"Cache miss for {article_link}, scraping now...")
        comments = scrape_comments(article_link)
        COMMENTS_CACHE[article_link] = comments
    else:
        print(f"Cache hit for {article_link}: {len(comments)} comments")
    
    if not comments:
        print("No comments found")
    
    sentiment_results, top_5_words = analyze_sentiment(comments)
    
    print("Top 5 Redundant Words:", top_5_words)
    
    return render_template('comments.html', 
                         sentiment_results=sentiment_results, 
                         top_5_words=top_5_words)


if __name__ == '__main__':
    app.run(debug=True)