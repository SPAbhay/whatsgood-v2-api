import os
import feedparser
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import create_client, Client
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from rss_feeds import FEEDS_TO_SCRAPE
import time
from datetime import datetime
from dateutil import parser as date_parser

# --- MODIFICATION ---
# ALL global variables are now set to None.
# The init phase will be lightning fast.
supabase: Client = None
pinecone_index = None
bi_encoder = None
model_path = os.path.join(os.path.dirname(__file__), 'model')
# --- END MODIFICATION ---


# HELPER FUNCTIONS
def clean_content(html_content):
    """Strips HTML tags and extra whitespace from content."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text()
    return " ".join(text.split())

def get_iso_date(published_str):
    """Converts various date formats to ISO 8601 string."""
    try:
        dt = date_parser.parse(published_str)
        return dt.isoformat()
    except (ValueError, TypeError):
        return datetime.now().isoformat()
    
# LAMBDA HANDLER
def ingest(event, context):
    """
    This is the main function that AWS Lambda will call.
    """
    
    # --- MODIFICATION ---
    # Use 'global' to access and initialize our clients ONCE.
    global supabase, pinecone_index, bi_encoder

    if supabase is None:
        print("Handler: Cold start. Initializing Supabase client...")
        url: str = os.environ.get("SUPABASE_URL")
        key: str = os.environ.get("SUPABASE_ANON_KEY")
        if not url or not key:
            raise ValueError("Error: SUPABASE_URL or SUPABASE_ANON_KEY not set.")
        supabase = create_client(url, key)
        print("Handler: Supabase client initialized.")

    if pinecone_index is None:
        print("Handler: Cold start. Initializing Pinecone connection...")
        PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        PINECONE_INDEX_NAME = "whats-good-v2"
        if not PINECONE_API_KEY:
            raise ValueError("Error: PINECONE_API_KEY not set.")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        print("Handler: Pinecone index obtained.")
    
    if bi_encoder is None:
        print("Handler: Cold start. Loading bi-encoder model: all-MiniLM-L6-v2...")
        bi_encoder = SentenceTransformer(model_path)
        print("Handler: Bi-encoder model loaded.")
    else:
        print("Handler: Warm start. All clients and model already loaded.")
    # --- END MODIFICATION ---

    print("Starting ingestion cycle...")
    new_articles_ingested = 0

    for industry, feed_list in FEEDS_TO_SCRAPE["feeds"].items():
        print(f"--- Parsing industry: {industry} ---")
        
        for feed_info in feed_list:
            feed_url = feed_info["url"]
            feed_name = feed_info["name"]
            
            print(f"Parsing feed: {feed_name} ({feed_url})")
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries:
                try:
                    article_url = entry.link
                    if not article_url:
                        continue

                    # --- A. Check for Duplicates ---
                    response = supabase.table('articles').select('id').eq('article_url', article_url).execute()
                    
                    if len(response.data) > 0:
                        continue

                    # --- B. Process New Article ---
                    print(f"NEW ARTICLE FOUND: {entry.title}")
                    
                    # 1. Get Metadata
                    title = entry.title
                    content_html = entry.get('summary', entry.get('content', [{}])[0].get('value', ''))
                    content = clean_content(content_html)
                    published_date = get_iso_date(entry.get('published', ''))

                    if not content:
                        print("Article has no content, skipping.")
                        continue

                    # 2. Create Bi-Encoder Embedding
                    vector = bi_encoder.encode(content).tolist() 
                    
                    # 3. Save to Supabase
                    new_article_data = {
                        'article_url': article_url,
                        'title': title,
                        'content': content,
                        'published_date': published_date,
                        'industry': industry  
                    }
                    insert_response = supabase.table('articles').insert(new_article_data).execute()
                    
                    if not insert_response.data:
                        print(f"Error inserting article to Supabase: {insert_response.error}")
                        continue
                        
                    new_article_id = insert_response.data[0]['id']
                    
                    # 4. Save to Pinecone
                    pinecone_index.upsert(
                        vectors=[(new_article_id, vector)],
                    )
                    
                    new_articles_ingested += 1
                    
                    time.sleep(0.5) 

                except Exception as e:
                    print(f"Error processing entry {entry.get('link', 'NO_LINK')}: {e}")

    print(f"--- Ingestion cycle complete. Ingested {new_articles_ingested} new articles. ---")
    return {
        'statusCode': 200,
        'body': f'Ingested {new_articles_ingested} new articles.'
    }
    
# LOCAL TESTING
if __name__ == "__main__":
    # Local testing will now also follow the same lazy-loading logic
    print("--- RUNNING INGESTION LOCALLY ---")
    
    # We must initialize clients here for local testing
    # Note: This is a simplified check.
    if pinecone_index is None:
        print("Local test: Initializing clients...")
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_ANON_KEY")
        supabase = create_client(url, key)
        
        PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
        PINECONE_INDEX_NAME = "whats-good-v2"
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        print("Local test: Clients initialized.")
    
    try:
        stats = pinecone_index.describe_index_stats()
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' initial state: {stats['total_vector_count']} vectors")
    except Exception as e:
        print(f"Could not get Pinecone stats: {e}")

    ingest(event=None, context=None)
    
    print("--- LOCAL RUN COMPLETE ---")
    
    try:
        stats = pinecone_index.describe_index_stats()
        print(f"Pinecone index '{PINECONE_INDEX_NAME}' final state: {stats['total_vector_count']} vectors")
    except Exception as e:
        print(f"Could not get Pinecone stats: {e}")