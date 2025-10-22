import os
import json
from supabase import create_client, Client

# --- Clients are loaded ONCE during the cold start ---
# This code runs during the "init" phase.
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(url, key)

def handler(event, context):
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