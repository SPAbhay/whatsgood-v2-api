import os
import json
from supabase import create_client, Client

# --- Clients are loaded ONCE during the cold start ---
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

def handler(event, context):
    """
    This is the new handler function that API Gateway will call.
    """
    try:
        # 1. Get the request data from the 'body'
        body = json.loads(event.get('body', '{}'))

        # 2. Your original Supabase logic
        # We can just pass the whole dictionary
        response = supabase.table('user_interactions').insert(body).execute()

        if not response.data:
            raise Exception("Failed to insert interaction. Check user_id or article_id.")

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
                "service": "interaction-service",
                "logged_interaction": response.data[0]
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': { 'Access-Control-Allow-Origin': '*' }, # CORS
            'body': json.dumps({"message": str(e)})
        }