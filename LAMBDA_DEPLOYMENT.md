# Lambda Deployment Guide for Ambulance RAG

## Problem
Bundle size (4980.79 MB) exceeds Lambda limit (500 MB). Main culprits:
- `sentence-transformers` + PyTorch (~400-500MB)
- `chromadb` with dependencies (~100-200MB)

## Solution: Use Lambda Layers

### Step 1: Create Layer for Heavy Dependencies
```bash
# Create directories
mkdir python
cd python

# Install heavy packages into python/ directory
pip install sentence-transformers chromadb -t .

# Zip it (max size per layer: 250MB, can use multiple layers)
cd ..
zip -r sentence-transformers-layer.zip python/
zip -r chromadb-layer.zip python/
```

### Step 2: Deploy Lambda Layers to AWS
```bash
aws lambda publish-layer-version \
  --layer-name sentence-transformers \
  --zip-file fileb://sentence-transformers-layer.zip \
  --compatible-runtimes python3.12

aws lambda publish-layer-version \
  --layer-name chromadb \
  --zip-file fileb://chromadb-layer.zip \
  --compatible-runtimes python3.12
```

### Step 3: Create Deployment Package (optimized)
```bash
# Use requirements-lambda.txt (lightweight deps only)
pip install -r requirements-lambda.txt -t ./package/

# Add your code
cp -r backend/ package/
cp app.py package/
cp lambda_handler.py package/

# Zip it
cd package
zip -r ../lambda-deployment.zip .
cd ..

# Upload to Lambda (note the size should now be <500MB)
aws lambda create-function \
  --function-name ambulance-rag \
  --runtime python3.12 \
  --role arn:aws:iam::YOUR_ACCOUNT_ID:role/lambda-role \
  --handler lambda_handler.handler \
  --zip-file fileb://lambda-deployment.zip \
  --layers arn:aws:lambda:REGION:ACCOUNT:layer:sentence-transformers:VERSION \
           arn:aws:lambda:REGION:ACCOUNT:layer:chromadb:VERSION \
  --timeout 60 \
  --memory-size 3008
```

### Step 4: Add Mangum to Requirements
The handler needs `mangum` to convert FastAPI to Lambda:
```bash
pip install mangum
echo "mangum==0.18.0" >> requirements-lambda.txt
```

## Alternative: Use API-Based Embeddings
Instead of local `sentence-transformers`, use embedding APIs:
- Groq embeddings (already used for LLM)
- OpenAI embeddings
- AWS Bedrock embeddings

This eliminates the largest dependency (~400MB saved).

## Ephemeral Storage
After deployment, Lambda gets 500MB at `/tmp/` for chromadb, model caches, etc.
Configure Chroma to use `/tmp/`:
```python
# In backend/models.py or rag_engine.py
persistent_client = chromadb.HttpClient(host="localhost", port=8000)
# OR
client = chromadb.PersistentClient(path="/tmp/chroma_data")
```

## Summary of Size Reductions
- Base code: ~5-10MB
- Lightweight deps: ~50-80MB
- Heavy deps via layers: isolated from 500MB limit
- Total deployment: <500MB ✓
