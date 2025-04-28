import boto3
from langchain_aws import ChatBedrock

# AWS Bedrock client
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

# Llama 3 70B Instruct model connection
llm = ChatBedrock(
    client=bedrock_runtime,
    model_id="meta.llama3-70b-instruct-v1:0",
    model_kwargs={
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 2048
    }
)
