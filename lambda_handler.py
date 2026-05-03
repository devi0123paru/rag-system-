"""
AWS Lambda handler for FastAPI app
Compatible with 500MB ephemeral storage limit
"""

from mangum import Mangum
from app import app

# Mangum converts FastAPI to ASGI Lambda handler
handler = Mangum(app, lifespan="off")
