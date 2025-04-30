import logging
import os
import boto3
from fastapi import FastAPI, Depends, HTTPException
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection
from models import ChatRequest, SearchRequest, SearchResponse, SearchResultItem
from chatbot_graph_sql_query import chatbot
from db import get_db
from sqlalchemy.ext.asyncio import AsyncSession

app = FastAPI()

# In-memory chat sessions (only for demo — in production use RDS!)
session_memory = {}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration from environment variables
AWS_REGION = os.getenv("AWS_REGION")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID")
OPENSEARCH_COLLECTION_ENDPOINT = os.getenv("OPENSEARCH_COLLECTION_ENDPOINT")
OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME")

# Basic validation for required environment variables
if not all([AWS_REGION, BEDROCK_MODEL_ID, OPENSEARCH_COLLECTION_ENDPOINT, OPENSEARCH_INDEX_NAME]):
    logger.error("Missing required environment variables (AWS_REGION, BEDROCK_MODEL_ID, OPENSEARCH_COLLECTION_ENDPOINT, OPENSEARCH_INDEX_NAME)")
    # In a real app, you might exit or raise a more specific configuration error
    exit(1)

# Initialize Bedrock Runtime Client
try:
    bedrock_runtime_client = boto3.client(
        service_name='bedrock-runtime',
        region_name=AWS_REGION
    )
    logger.info(f"Bedrock client initialized for region {AWS_REGION}")
except Exception as e:
    logger.exception(f"Failed to initialize Bedrock client: {e}")
    # Depending on the app's requirements, you might want to exit or handle this differently
    exit(1)

# Initialize OpenSearch Client using SigV4 Auth
try:
    credentials = boto3.Session().get_credentials()
    # Ensure credentials are valid before proceeding
    if credentials is None or credentials.access_key is None:
         raise ValueError("AWS credentials not found. Configure via AWS CLI, Env Vars, or IAM Role.")

    auth = AWSV4SignerAuth(credentials, AWS_REGION, 'aoss') # Use 'aoss' for Serverless
    opensearch_client = OpenSearch(
        hosts=[{'host': OPENSEARCH_COLLECTION_ENDPOINT, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20
    )
    # Test connection
    if not opensearch_client.ping():
        logger.warning("Failed to ping OpenSearch Serverless Collection. Check endpoint and permissions.")
    else:
        logger.info(f"OpenSearch client initialized for endpoint {OPENSEARCH_COLLECTION_ENDPOINT}")
except Exception as e:
    logger.exception(f"Failed to initialize OpenSearch client: {e}")
    # Depending on the app's requirements, you might want to exit or handle this differently
    exit(1)

# --- FastAPI App and Models ---

app = FastAPI(
    title="Document Search API",
    description="API to search relevant documents based on user query using vector similarity."
)



# --- Helper Functions ---

def get_embedding_from_bedrock(text_input: str) -> list[float]:
    """Calls Bedrock to get embedding for the given text."""
    if not text_input.strip():
        raise ValueError("Input text cannot be empty")

    try:
        body = json.dumps({"inputText": text_input})
        response = bedrock_runtime_client.invoke_model(
            body=body,
            modelId=BEDROCK_MODEL_ID,
            accept='application/json',
            contentType='application/json'
        )
        response_body = json.loads(response['body'].read())
        embedding = response_body.get('embedding')
        if embedding is None:
            raise ValueError("Embedding not found in Bedrock response")
        logger.info(f"Successfully generated embedding for query (first 50 chars): {text_input[:50]}...")
        return embedding
    except Exception as e:
        logger.exception(f"Error getting embedding from Bedrock for text '{text_input[:50]}...': {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding from Bedrock.")

def search_documents_in_opensearch(query_embedding: list[float], k: int, source_filter: str | None = None) -> list[SearchResultItem]:
    """Performs k-NN search in OpenSearch Serverless."""

    knn_query = {
        "vector": query_embedding,
        "k": k
    }

    # Base query structure for k-NN search
    query_body = {
        "size": k,
        "_source": ["text", "source_file"], # Specify fields to retrieve
        "query": {
            # If no filter, the query is just the knn part
            "knn": {
                "text_embedding": knn_query # Field name from your mapping
            }
        }
    }

    # If a source filter is provided, wrap the knn query in a boolean query
    if source_filter:
        query_body["query"] = {
            "bool": {
                "filter": [
                    { "term": { "source_file": source_filter } } # Filter by source_file
                ],
                "must": [ # knn query must still match
                    {
                        "knn": {
                            "text_embedding": knn_query
                        }
                    }
                ]
            }
        }
        logger.info(f"Performing k-NN search with k={k} and filter source='{source_filter}'")
    else:
         logger.info(f"Performing k-NN search with k={k} (no filter)")


    try:
        response = opensearch_client.search(
            index=OPENSEARCH_INDEX_NAME,
            body=query_body
        )

        results = []
        hits = response.get('hits', {}).get('hits', [])
        for hit in hits:
            source_data = hit.get('_source', {})
            text = source_data.get('text')
            source_file = source_data.get('source_file')
            score = hit.get('_score', 0.0) # Similarity score

            if text and source_file: # Ensure necessary fields are present
                 results.append(SearchResultItem(source_file=source_file, text=text, score=score))

        logger.info(f"OpenSearch query returned {len(results)} results.")
        return results

    except Exception as e:
        logger.exception(f"Error searching documents in OpenSearch: {e}")
        # You might want to inspect the error type for more specific HTTP codes
        raise HTTPException(status_code=500, detail="Failed to search documents in OpenSearch.")


# --- API Endpoint ---



@app.post("/chat/{session_id}", response_model=SearchResponse)
async def search_endpoint(session_id: int, request: SearchRequest, db: AsyncSession = Depends(get_db)):
    logger.info(f"Received search request: query='{request.query[:50]}...', top_k={request.top_k}, filter='{request.filter_source}'")
    memory = session_memory.get(session_id, [])
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        query_embedding = get_embedding_from_bedrock(request.query)
    except ValueError as ve: # Handle specific case of empty input before Bedrock call
         raise HTTPException(status_code=400, detail=str(ve))
    
    search_results_instruction = search_documents_in_opensearch(
        query_embedding=query_embedding,
        k=request.top_k,
        source_filter="instruction" # Pass the optional filter
    )

    search_results_domain_knowledge = search_documents_in_opensearch(
        query_embedding= query_embedding,
        k= 1, #request.top_k,
        source_filter= "domain_knowledge" # Pass the optional filter
    )

    instruction_content= search_results_instruction[0]["source"]["content"]
    domain_knowledge_content= search_results_domain_knowledge[0]["source"]["content"]


    msg_dict = {
        "domain": domain_knowledge_content,
        "instruction": instruction_content,
        "schema": "dummy",
        "query": request.query
    }

    msg = f"[Domain]: {msg_dict['domain']}\n[Schema]: {msg_dict['schema']}\n[Instruction]: {msg_dict['instruction']}\n[Query]: {msg_dict['query']}"

    result = chatbot.invoke({
        "user_message": msg,
        "chat_history": memory
    })

    session_memory[session_id] = result["chat_history"]

    sql_query = result["bot_response"]
    result = db.execute(sql_query)

    result_string = ""

    for row in result:
        result_string = result_string + "/n" + str(dict(row)) 

    final_result = chatbot.invoke({
        "user_message": result_string,
        "chat_history": memory
    })



    return {"response": result["bot_response"]}
