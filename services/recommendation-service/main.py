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
import json
import traceback

# --- 1. INITIALIZATION & MODEL LOADING ---

# Global clients and models
supabase: Client = None
pinecone_index = None
groq_client: Groq = None
bi_encoder: SentenceTransformer = None
cross_encoder: CrossEncoder = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup/shutdown logic.
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
    
    # Load BI-ENCODER model
    bi_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Load CROSS-ENCODER model
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    
    print("Models and clients loaded successfully.")
    
    yield # The application runs here
    
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. HELPER FUNCTIONS ---

def get_freshness_score(published_date_str: str) -> float:
    """
    Calculates a freshness score (0.0 to 1.0).
    """
    try:
        published_date = datetime.fromisoformat(published_date_str).replace(tzinfo=timezone.utc)
        days_old = (datetime.now(timezone.utc) - published_date).days
        
        if days_old < 0: return 1.0
        if days_old > 14: return 0.0
        
        return 1.0 - (days_old / 14.0)
    except:
        return 0.5

async def get_dynamic_query_vector(user_id: str) -> np.ndarray:
    """
    Creates a dynamic query vector based on persona and recent interactions.
    """
    # 1. Get base persona vector
    user_response = supabase.table('users').select('base_persona').eq('id', user_id).execute()
    if not user_response.data:
        raise HTTPException(status_code=404, detail="User not found")
    
    base_persona_text = user_response.data[0]['base_persona']

    # --- THIS IS THE FIX ---
    if not base_persona_text:
        print(f"User {user_id} has no persona, using default for vector.")
        base_persona_text = "A person interested in technology, finance, and world news."
    # --- END FIX ---
    
    base_vector = bi_encoder.encode(base_persona_text)
    
    # 2. Get recent interactions
    interaction_response = supabase.table('user_interactions').select('article_id, interaction_type').eq('user_id', user_id).order('created_at', desc=True).limit(10).execute()
    
    if not interaction_response.data:
        return base_vector

    # 3. Fetch vectors for liked/disliked articles
    article_ids = [item['article_id'] for item in interaction_response.data]
    fetch_response = pinecone_index.fetch(ids=article_ids)
    fetched_vectors = {vec_id: vec.values for vec_id, vec in fetch_response.vectors.items()}

    # 4. Vector Arithmetic
    dynamic_vector = base_vector
    weight = 0.2

    for item in interaction_response.data:
        article_id = item['article_id']
        interaction_type = item['interaction_type']
        
        if article_id in fetched_vectors:
            article_vector = np.array(fetched_vectors[article_id])
            
            if interaction_type == 'like':
                dynamic_vector = dynamic_vector + weight * (article_vector - dynamic_vector)
            elif interaction_type == 'dislike':
                dynamic_vector = dynamic_vector - weight * (article_vector - dynamic_vector)
    
    return dynamic_vector / np.linalg.norm(dynamic_vector)


# --- 3. PYDANTIC MODELS (for response) ---

class RecommendedArticle(BaseModel):
    id: str
    title: str
    url: str
    summary: str
    reason: str  # <--- THIS IS NEW
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
    The main multi-stage RAG pipeline endpoint
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
            include_metadata=False
        )
        initial_retrieved_ids = [match.id for match in retrieve_response.matches]
        pinecone_scores = {match.id: match.score for match in retrieve_response.matches}

        print(f"RecSvc: Stage 2.5 - Filtering out interacted articles for user {user_id}...")
        try:
            # Get ALL article IDs this user has ever liked or disliked
            interacted_response = supabase.table('user_interactions') \
                .select('article_id') \
                .eq('user_id', user_id) \
                .execute()

            interacted_ids = {item['article_id'] for item in interacted_response.data} if interacted_response.data else set()
            print(f"RecSvc: User has interacted with {len(interacted_ids)} articles.")

            # Filter the retrieved list (Use initial_retrieved_ids here)
            retrieved_ids = [id for id in initial_retrieved_ids if id not in interacted_ids]
            print(f"RecSvc: Filtered down to {len(retrieved_ids)} candidates.")

            # Ensure we still have enough candidates for the next stages
            if len(retrieved_ids) < 5:
                 print("RecSvc: Warning - Filtering removed too many candidates. Need at least 5 for generation.")
                 # Fallback: If ALL were filtered, return empty now to avoid errors later
                 if not retrieved_ids:
                     print("RecSvc: No candidates remaining after filtering.")
                     return RecommendationResponse(
                         service="recommendation-service", user_id=user_id, recommendations=[]
                     )
                 # If some remain (1-4), we proceed but will get fewer than 5 recs.

        except Exception as filter_err:
            print(f"RecSvc: Error filtering interacted articles: {filter_err}. Proceeding without filtering.")
            retrieved_ids = initial_retrieved_ids

        if not initial_retrieved_ids:
            return RecommendationResponse(
                service="recommendation-service",
                user_id=user_id,
                recommendations=[]
            )

        # --- Stage 3: Fetch Metadata ---
        ids_to_fetch = retrieved_ids[:50]
        print(f"RecSvc: Stage 3 - Fetching metadata for up to {len(ids_to_fetch)} articles from Supabase...") # Updated print
        articles_response = supabase.table('articles').select('*').in_('id', ids_to_fetch).execute() # Use ids_to_fetch
        if not articles_response.data:
            print("RecSvc: No metadata found for remaining candidates.")
            # Use the Pydantic model for the empty response
            return RecommendationResponse(
                 service="recommendation-service", user_id=user_id, recommendations=[]
            )

        # --- Stage 4: Re-rank (Accurate & Slow) ---
        print("RecSvc: Stage 4 - Re-ranking with Cross-Encoder and Time Decay...")

        # --- THIS IS THE SECOND FIX ---
        user_response = supabase.table('users').select('base_persona').eq('id', user_id).execute()
        if not user_response.data:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found in Stage 4.")
        
        user_persona_text = user_response.data[0]['base_persona']
        if not user_persona_text:
            print(f"User {user_id} has no persona, using default for re-ranking.")
            user_persona_text = "A person interested in technology, finance, and world news."
        # --- END FIX ---

        cross_encoder_pairs = []
        for article in articles_response.data:
            cross_encoder_pairs.append([user_persona_text, article['content']])

        cross_scores = cross_encoder.predict(cross_encoder_pairs)

        ranked_list = []
        for i, article in enumerate(articles_response.data):
            article_id = article['id']

            p_score = pinecone_scores.get(article_id, 0.0)
            c_score = cross_scores[i]
            c_score_normalized = 1 / (1 + np.exp(-c_score))
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

        # --- NEW PROMPT ---
        system_prompt = "You are a sharp news curator writing directly to a user. Your style is concise, impactful, and insightful, like a high-quality news digest. You avoid filler words and get straight to the point."
        human_prompt = f"""
My user's persona is: "{user_persona_text}"

Based *only* on the user's persona and the 5 articles provided below, do the following:

1.  For each article, write a single, **crisp "summary" sentence (max 20-25 words)**. This sentence must capture the absolute core takeaway or most impactful point of the article. It should subtly align with the user's interests but feel like a standalone, objective insight. **Do not mention the user or their persona.** Avoid hyphens and robotic phrasing. Use strong verbs.
2.  For each article, write a "reason" sentence (1-2 sentences). This is an internal justification explaining the specific link between the article's core takeaway and the user's persona.
3.  Return your response as a single, valid JSON array.
4.  The JSON array must contain 5 objects. Each object must have *only* these keys: "id", "title", "summary", "reason".
5.  Use the exact "id" and "title" provided for each article in the context.

**Examples of desired crisp summary style:**

* *Article about cricket redemption:* "Mental resilience proves key as cricket teams navigate the high-stakes pressure of seeking redemption after major setbacks."
* *Article about new MLOps tool:* "A new open-source framework aims to significantly simplify the deployment pipeline for large language models."
* *Article about sustainable travel in Europe:* "Mindful travel choices are gaining momentum, potentially reshaping the future of tourism across popular European destinations."

**Here is the context for the 5 articles you need to process:**
{context_block}
"""
        # --- END NEW PROMPT ---

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        raw_llm_output = chat_completion.choices[0].message.content
        print(f"RecSvc: Raw LLM Output:\n---\n{raw_llm_output}\n---") 

        llm_response_json = None
        try:
            llm_response_json = json.loads(raw_llm_output)
        except json.JSONDecodeError as json_err:
            print(f"RecSvc: FAILED TO PARSE LLM JSON: {json_err}")
            raise HTTPException(status_code=500, detail="LLM returned invalid JSON format.")

        parsed_list = None
        if isinstance(llm_response_json, list):
            parsed_list = llm_response_json
        elif isinstance(llm_response_json, dict) and len(llm_response_json) == 1:
             key = list(llm_response_json.keys())[0]
             if isinstance(llm_response_json[key], list):
                 print(f"RecSvc: Found list inside dictionary key '{key}'. Using that.")
                 parsed_list = llm_response_json[key]

        if parsed_list is None:
             print(f"RecSvc: LLM JSON was not a list or dict containing a list. Type: {type(llm_response_json)}")
             raise HTTPException(status_code=500, detail="LLM JSON response format is incorrect (expected a list).")

        llm_summaries = {}
        try:
            if parsed_list and isinstance(parsed_list[0], dict) and 'id' in parsed_list[0]:
                 llm_summaries = {item['id']: item for item in parsed_list}
            elif not parsed_list:
                 print("RecSvc: LLM returned an empty list of summaries.")
                 pass
            else:
                 print(f"RecSvc: Items in the parsed list are not dictionaries or missing 'id'. First item: {parsed_list[0]}")
                 raise HTTPException(status_code=500, detail="LLM JSON list items are not in the expected format.")
        except KeyError as key_err:
             print(f"RecSvc: Missing expected key in LLM JSON item: {key_err}")
             raise HTTPException(status_code=500, detail=f"LLM JSON item missing key: {key_err}")
        except TypeError as type_err:
             print(f"RecSvc: TypeError during summary processing: {type_err}")
             raise HTTPException(status_code=500, detail="LLM response processing error (TypeError).")


        # Build the final response list
        final_response_articles = []
        for article in final_top_5_articles:
            article_id = article['id']
            summary_data = llm_summaries.get(article_id)

            if summary_data:
                if 'title' in summary_data and 'summary' in summary_data:
                    final_response_articles.append(RecommendedArticle(
                        id=article_id,
                        title=summary_data['title'],
                        url=article['article_url'], 
                        summary=summary_data['summary'],
                        reason=summary_data['reason'], 
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
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

@app.get("/") # Health check
def read_root():
    if cross_encoder and groq_client:
        return {"service": "recommendation-service", "status": "running", "models_loaded": True}
    return {"service": "recommendation-service", "status": "initializing", "models_loaded": False}