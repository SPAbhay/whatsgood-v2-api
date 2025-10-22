import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
from pinecone import Pinecone
from groq import Groq
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from datetime import datetime, timezone

# --- 1. INITIALIZATION & MODEL LOADING ---

# Load .env file from the *root* directory
# dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
# load_dotenv(dotenv_path=dotenv_path)

# Global clients and models
# These will be initialized on startup using the 'lifespan' manager
supabase: Client = None
pinecone_index = None
groq_client: Groq = None
bi_encoder: SentenceTransformer = None
cross_encoder: CrossEncoder = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This is the modern way to handle startup/shutdown logic in FastAPI.
    It's superior to @app.on_event("startup")
    """
    global supabase, pinecone_index, groq_client, bi_encoder, cross_encoder
    
    print("Recommendation Service: Initializing clients and loading models...")
    
    # Initialize Supabase
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_ANON_KEY")
    supabase = create_client(url, key)
    
    # Initialize Pinecone
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = "whats-good-v2"
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(PINECONE_INDEX_NAME)

    # Initialize Groq
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    # Load BI-ENCODER model (for embedding the persona)
    # This MUST be the same model used in the ingestion-service
    bi_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Load CROSS-ENCODER model (for re-ranking)
    # This model is much more accurate for relevance scoring
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    
    print("Models and clients loaded successfully.")
    
    yield # The application runs here
    
    # Shutdown logic (if any) can go here
    print("Recommendation Service: Shutting down.")

# Pass the lifespan manager to the FastAPI app
app = FastAPI(lifespan=lifespan)

origins = [
    "*", 
    "https://whatsgood.abhay-arora.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # Usually needed
    allow_methods=["*"],    # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],    # Allow all headers
)

# --- 2. HELPER FUNCTIONS ---

def get_freshness_score(published_date_str: str) -> float:
    """
    Calculates a freshness score (0.0 to 1.0).
    Articles from today get 1.0. Score decays over 14 days.
    """
    try:
        published_date = datetime.fromisoformat(published_date_str).replace(tzinfo=timezone.utc)
        days_old = (datetime.now(timezone.utc) - published_date).days
        
        if days_old < 0: return 1.0 # Future posts
        if days_old > 14: return 0.0 # Older than 2 weeks
        
        # Linear decay: 1.0 - (days_old / 14)
        return 1.0 - (days_old / 14.0)
    except:
        return 0.5 # Default score if date is malformed

async def get_dynamic_query_vector(user_id: str) -> np.ndarray:
    """
    Creates a dynamic query vector based on persona and recent interactions.
    """
    # 1. Get base persona vector
    user_response = supabase.table('users').select('base_persona').eq('id', user_id).execute()
    if not user_response.data:
        raise HTTPException(status_code=404, detail="User not found")
    
    base_persona_text = user_response.data[0]['base_persona']
    base_vector = bi_encoder.encode(base_persona_text)
    
    # 2. Get recent interactions
    interaction_response = supabase.table('user_interactions').select('article_id, interaction_type').eq('user_id', user_id).order('created_at', desc=True).limit(10).execute()
    
    if not interaction_response.data:
        return base_vector # No interactions, return base vector

    # 3. Fetch vectors for liked/disliked articles
    article_ids = [item['article_id'] for item in interaction_response.data]
    
    # Use Pinecone's fetch API
    fetch_response = pinecone_index.fetch(ids=article_ids)
    
    # Create a simple lookup dictionary for fetched vectors
    fetched_vectors = {vec_id: vec.values for vec_id, vec in fetch_response.vectors.items()}

    # 4. Vector Arithmetic: Adjust the base vector
    # This is the "Dynamic Persona" logic
    dynamic_vector = base_vector
    weight = 0.2 # How much each interaction adjusts the vector

    for item in interaction_response.data:
        article_id = item['article_id']
        interaction_type = item['interaction_type']
        
        if article_id in fetched_vectors:
            article_vector = np.array(fetched_vectors[article_id])
            
            if interaction_type == 'like':
                dynamic_vector = dynamic_vector + weight * (article_vector - dynamic_vector)
            elif interaction_type == 'dislike':
                dynamic_vector = dynamic_vector - weight * (article_vector - dynamic_vector)
    
    # Normalize the final vector (good practice)
    return dynamic_vector / np.linalg.norm(dynamic_vector)


# --- 3. PYDANTIC MODELS (for response) ---

class RecommendedArticle(BaseModel):
    id: str
    title: str
    url: str
    summary: str
    published_date: str
    industry: str
    final_score: float

class RecommendationResponse(BaseModel):
    service: str
    user_id: str
    recommendations: list[RecommendedArticle]

# --- 4. THE MAIN RECOMMENDATION ENDPOINT ---

@app.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(user_id: str):
    """
    The main multi-stage RAG pipeline endpoint.
    """
    try:
        print(f"RecSvc: Received request for user_id: {user_id}")

        # --- Stage 1: Get Dynamic Query Vector ---
        print("RecSvc: Stage 1 - Generating dynamic query vector...")
        query_vector = await get_dynamic_query_vector(user_id)

        # --- Stage 2: Retrieve (Fast & Broad) ---
        print("RecSvc: Stage 2 - Retrieving Top 50 candidates from Pinecone...")
        retrieve_response = pinecone_index.query(
            vector=query_vector.tolist(),
            top_k=50,
            include_metadata=False # We only need IDs
        )
        retrieved_ids = [match.id for match in retrieve_response.matches] # Use attribute access
        pinecone_scores = {match.id: match.score for match in retrieve_response.matches} # Use attribute access

        if not retrieved_ids:
            return RecommendationResponse(
                service="recommendation-service",
                user_id=user_id,
                recommendations=[]
            )

        # --- Stage 3: Fetch Metadata ---
        print(f"RecSvc: Stage 3 - Fetching metadata for {len(retrieved_ids)} articles from Supabase...")
        articles_response = supabase.table('articles').select('*').in_('id', retrieved_ids).execute()

        if not articles_response.data:
            raise HTTPException(status_code=404, detail="No article metadata found for retrieved IDs")

        # --- Stage 4: Re-rank (Accurate & Slow) ---
        print("RecSvc: Stage 4 - Re-ranking with Cross-Encoder and Time Decay...")

        user_persona_text = supabase.table('users').select('base_persona').eq('id', user_id).execute().data[0]['base_persona']

        cross_encoder_pairs = []
        for article in articles_response.data:
            cross_encoder_pairs.append([user_persona_text, article['content']])

        cross_scores = cross_encoder.predict(cross_encoder_pairs)

        ranked_list = []
        for i, article in enumerate(articles_response.data):
            article_id = article['id']

            p_score = pinecone_scores.get(article_id, 0.0)
            c_score = cross_scores[i]
            c_score_normalized = 1 / (1 + np.exp(-c_score)) # Sigmoid
            f_score = get_freshness_score(article['published_date'])

            final_score = (
                (0.60 * c_score_normalized) +
                (0.30 * p_score) +
                (0.10 * f_score)
            )

            article['final_score'] = final_score
            ranked_list.append(article)

        ranked_list.sort(key=lambda x: x['final_score'], reverse=True)
        final_top_5_articles = ranked_list[:5]

        # --- Stage 5: Generate (Summarize) ---
        print("RecSvc: Stage 5 - Generating summaries for Top 5 with Groq...")

        context_block = ""
        for i, article in enumerate(final_top_5_articles):
            context_block += f"--- ARTICLE {i+1} (ID: {article['id']}) ---\n"
            context_block += f"Title: {article['title']}\n"
            context_block += f"Content: {article['content'][:1500]}...\n\n"

        system_prompt = "You are a professional, world-class news analyst."
        human_prompt = f"""
My personal profile/persona is: "{user_persona_text}"

Based *only* on my persona and the context from the 5 articles provided below, please do the following:
1.  For each of the 5 articles, write a 2-3 sentence, crisp summary explaining *why* it is relevant to me.
2.  Return your response as a single, valid JSON array, with no other text before or after it.
3.  The JSON array should contain 5 objects. Each object must have *only* these keys: "id", "title", "summary".
4.  Use the exact "id" and "title" provided for each article in the context.

Here is the context:
{context_block}
"""

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt}
            ],
            model="llama-3.1-8b-instant", # Use the correct, available model
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        # --- Debug and Parse Robustly ---
        raw_llm_output = chat_completion.choices[0].message.content
        print(f"RecSvc: Raw LLM Output:\n---\n{raw_llm_output}\n---") # DEBUG PRINT

        import json
        llm_response_json = None # Initialize
        try:
            llm_response_json = json.loads(raw_llm_output)
        except json.JSONDecodeError as json_err:
            print(f"RecSvc: FAILED TO PARSE LLM JSON: {json_err}")
            print(f"RecSvc: Raw output was: {raw_llm_output}")
            raise HTTPException(status_code=500, detail="LLM returned invalid JSON format.")

        # Ensure the parsed result is a list or a dict containing a list
        parsed_list = None
        if isinstance(llm_response_json, list):
            parsed_list = llm_response_json
        elif isinstance(llm_response_json, dict) and len(llm_response_json) == 1:
             # Handle cases where json_object mode wraps the list, e.g., {"summaries": [...]}
             key = list(llm_response_json.keys())[0]
             if isinstance(llm_response_json[key], list):
                 print(f"RecSvc: Found list inside dictionary key '{key}'. Using that.")
                 parsed_list = llm_response_json[key]

        if parsed_list is None:
             print(f"RecSvc: LLM JSON was not a list or dict containing a list. Type: {type(llm_response_json)}")
             print(f"RecSvc: Parsed content: {llm_response_json}")
             raise HTTPException(status_code=500, detail="LLM JSON response format is incorrect (expected a list).")
        # --- End Debug/Parse ---

        # Now, `parsed_list` should be the list you expect
        llm_summaries = {}
        try:
            # Check list isn't empty and items are dicts with 'id'
            if parsed_list and isinstance(parsed_list[0], dict) and 'id' in parsed_list[0]:
                 llm_summaries = {item['id']: item for item in parsed_list}
            elif not parsed_list: # Handle empty list case
                 print("RecSvc: LLM returned an empty list of summaries.")
                 # Decide how to handle this - maybe return empty recommendations?
                 # For now, we'll continue and the loop below will just find nothing.
                 pass
            else: # Items are not dicts or missing 'id'
                 print(f"RecSvc: Items in the parsed list are not dictionaries or missing 'id'. First item: {parsed_list[0]}")
                 raise HTTPException(status_code=500, detail="LLM JSON list items are not in the expected format.")
        except KeyError as key_err:
             print(f"RecSvc: Missing expected key in LLM JSON item: {key_err}")
             print(f"RecSvc: Faulty item structure: {parsed_list[0] if parsed_list else 'None'}")
             raise HTTPException(status_code=500, detail=f"LLM JSON item missing key: {key_err}")
        except TypeError as type_err: # Catch the original error source
             print(f"RecSvc: TypeError during summary processing: {type_err}")
             print(f"RecSvc: Parsed list content: {parsed_list}")
             raise HTTPException(status_code=500, detail="LLM response processing error (TypeError).")


        # Build the final response list
        final_response_articles = []
        for article in final_top_5_articles:
            article_id = article['id']
            summary_data = llm_summaries.get(article_id)

            if summary_data:
                # Basic check for expected keys in summary_data
                if 'title' in summary_data and 'summary' in summary_data:
                    final_response_articles.append(RecommendedArticle(
                        id=article_id,
                        title=summary_data['title'],
                        url=article['article_url'],
                        summary=summary_data['summary'],
                        published_date=article['published_date'],
                        industry=article['industry'],
                        final_score=article['final_score']
                    ))
                else:
                     print(f"RecSvc: Warning - LLM summary for article {article_id} missing 'title' or 'summary'. Skipping.")


        print("RecSvc: Successfully completed recommendation pipeline.")
        return RecommendationResponse(
            service="recommendation-service",
            user_id=user_id,
            recommendations=final_response_articles
        )

    except Exception as e:
        print(f"RecSvc: CRITICAL ERROR - {e}")
        import traceback
        traceback.print_exc() # Print full stack trace to Docker logs
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

@app.get("/") # Health check
def read_root():
    # Check if models are loaded
    if cross_encoder and groq_client:
        return {"service": "recommendation-service", "status": "running", "models_loaded": True}
    return {"service": "recommendation-service", "status": "initializing", "models_loaded": False}