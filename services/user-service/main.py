import os
import json
import hashlib
from supabase import create_client, Client

# --- Clients are loaded ONCE during the cold start ---
# This code runs during the "init" phase.
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

def hash_key(api_key):
    """Creates a SHA-256 hash of the API key."""
    return hashlib.sha256(api_key.encode()).hexdigest()

def login_handler(event, context):
    """
    Handles the initial login.
    Finds a user by their API key or creates a new user.
    """
    try:
        body = json.loads(event.get('body', '{}'))
        api_key = body.get('api_key')
        if not api_key:
            raise ValueError("API Key is required")
            
        # 1. Hash the key for secure comparison
        key_hash = hash_key(api_key)
        
        # 2. Check if this user (key_hash) already exists
        response = supabase.table('users').select('*').eq('api_key_hash', key_hash).execute()
        
        user_data = None
        
        if response.data:
            # --- User Exists ---
            print("User found. Returning existing data.")
            user_data = response.data[0]
        else:
            # --- New User ---
            # This is their first login. Create a new user entry.
            print("First login for this key. Creating new user.")
            insert_response = supabase.table('users').insert({
                'api_key_hash': key_hash
                # We'll let them set the persona in the next step
            }).execute()
            
            if not insert_response.data:
                 raise Exception("Failed to create new user")
            user_data = insert_response.data[0]

        return {
            'statusCode': 200,
            'headers': { 'Access-Control-Allow-Origin': '*' },
            'body': json.dumps({
                "status": "ok",
                "user_id": user_data['id'],
                "base_persona": user_data.get('base_persona') # Will be null for new users
            })
        }

    except Exception as e:
        return { 'statusCode': 500, 'headers': { 'Access-Control-Allow-Origin': '*' }, 'body': json.dumps({"message": str(e)}) }

def persona_handler(event, context):
    """
    This is the new handler function that API Gateway will call.
    'event' is a dictionary containing the request info.
    # adding this line to trigger the workflow
    """
    try:
        # 1. Get the request data from the 'body'
        # The body comes in as a JSON string, so we must parse it
        body = json.loads(event.get('body', '{}'))
        
        user_id = body.get('user_id')
        persona = body.get('persona')

        if not user_id or not persona:
            raise ValueError("user_id and persona are required")

        # 2. Your original Supabase logic (this is unchanged)
        response = supabase.table('users').update({
            'base_persona': persona
        }).eq('id', user_id).execute()
        
        if not response.data:
             raise Exception("User not found")

        # 3. Return a special dictionary that API Gateway understands
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*', # Required for CORS
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST,OPTIONS'
            },
            'body': json.dumps({
                "status": "ok", 
                "service": "user-service",
                "user_id": user_id,
                "updated_persona": response.data[0]['base_persona']
            })
        }

    except Exception as e:
        # If anything goes wrong, return a 500 error
        return {
            'statusCode': 500,
            'headers': { 'Access-Control-Allow-Origin': '*' }, # CORS
            'body': json.dumps({"message": str(e)})
        }

def handler(event, context):
    """
    This is the main handler that routes to the correct function
    based on the HTTP method and path.
    """
    http_method = event.get('requestContext', {}).get('http', {}).get('method')
    path = event.get('requestContext', {}).get('http', {}).get('path')

    if http_method == 'POST' and path == '/v1/persona':
        return persona_handler(event, context) # Your existing persona handler
    
    if http_method == 'POST' and path == '/v1/login':
        return login_handler(event, context)
    
    # Default fallback
    return { 'statusCode': 404, 'body': json.dumps("Not Found") }