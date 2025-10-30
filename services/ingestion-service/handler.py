import os
import feedparser
import numpy as np
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import create_client, Client
from pinecone import Pinecone, ServerlessSpec
# from sentence_transformers import SentenceTransformer
from newspaper import Article
from datetime import datetime, timedelta, timezone
from rss_feeds import FEEDS_TO_SCRAPE
import time
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
        from sentence_transformers import SentenceTransformer
        print("Handler: Cold start. Loading bi-encoder model: all-MiniLM-L6-v2...")
        bi_encoder = SentenceTransformer(model_path)
        print("Handler: Bi-encoder model loaded.")
    else:
        print("Handler: Warm start. All clients and model already loaded.")
    # --- END MODIFICATION ---

    print("Starting ingestion cycle...")
    new_articles_ingested = 0
    skipped_future = 0
    skipped_old = 0
    skipped_failed = 0

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

                    # --- A. Check for Duplicates (Unchanged) ---
                    response = supabase.table('articles').select('id').eq('article_url', article_url).execute()
                    if len(response.data) > 0:
                        continue

                    # --- B. Process New Article (THE NEW LOGIC) ---
                    # Use newspaper3k to download and parse the full article
                    article = Article(article_url)
                    article.download()
                    article.parse()
                    
                    # 1. Get Metadata (Now reliable)
                    content = article.text
                    title = article.title
                    published_date_dt = article.publish_date

                    # We MUST have content and a title
                    if not content or not title:
                        print(f"Failed to parse content/title, skipping: {article_url}")
                        skipped_failed += 1
                        continue

                    # 2. Fix Date Issues (Your request)
                    if not published_date_dt:
                        # Fallback to feed date if newspaper fails
                        published_date_dt = date_parser.parse(entry.get('published', ''))
                    
                    # Ensure timezone-aware
                    if published_date_dt.tzinfo is None:
                        published_date_dt = published_date_dt.replace(tzinfo=timezone.utc)

                    # --- Date Guardrails ---
                    now = datetime.now(timezone.utc)
                    if published_date_dt > (now + timedelta(days=1)):
                        print(f"Article has future date, skipping: {title}")
                        skipped_future += 1
                        continue
                        
                    if published_date_dt < (now - timedelta(days=30)):
                        print(f"Article older than 30 days, skipping: {title}")
                        skipped_old += 1
                        continue
                    
                    published_date_iso = published_date_dt.isoformat()
                    # --- End Date Fixes ---

                    print(f"NEW ARTICLE FOUND: {title}")

                    # 3. Create Bi-Encoder Embedding (Unchanged)
                    vector = bi_encoder.encode(content).tolist() 
                    
                    # 4. Save to Supabase (Unchanged)
                    new_article_data = {
                        'article_url': article_url,
                        'title': title,
                        'content': content,
                        'published_date': published_date_iso,
                        'industry': industry  
                    }
                    insert_response = supabase.table('articles').insert(new_article_data).execute()
                    
                    if not insert_response.data:
                        print(f"Error inserting article to Supabase: {insert_response.error}")
                        continue
                        
                    new_article_id = insert_response.data[0]['id']
                    
                    # 5. Save to Pinecone (Unchanged)
                    pinecone_index.upsert(
                        vectors=[(new_article_id, vector)],
                    )
                    
                    new_articles_ingested += 1
                    time.sleep(0.5) # Unchanged

                except Exception as e:
                    print(f"Error processing entry {entry.get('link', 'NO_LINK')}: {e}")
                    skipped_failed += 1

    print(f"--- Ingestion cycle complete. ---")
    print(f"Ingested: {new_articles_ingested} new articles.")
    print(f"Skipped (Future Date): {skipped_future}")
    print(f"Skipped (Too Old): {skipped_old}")
    print(f"Skipped (Failed Parse): {skipped_failed}")
    
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