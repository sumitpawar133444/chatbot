from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    message: str

# Pydantic model for the request body
class SearchRequest(BaseModel):
    query: str = Field(..., example="How do I configure network access?")
    top_k: int = Field(5, gt=0, le=20, example=3) # Number of results to return

# Pydantic model for a single search result item
class SearchResultItem(BaseModel):
    source_file: str
    text: str
    score: float # Similarity score from OpenSearch

# Pydantic model for the response
class SearchResponse(BaseModel):
    results: list[SearchResultItem]