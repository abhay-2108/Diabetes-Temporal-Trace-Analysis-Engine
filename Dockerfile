# Stage 1: Build the React frontend
FROM node:20-alpine AS build-stage
WORKDIR /app/frontend

# Copy frontend dependency files
COPY src/web_app/frontend/package*.json ./
RUN npm install

# Copy all frontend files and build
COPY src/web_app/frontend/ ./
RUN npm run build


# Stage 2: Build the Python backend and serve
FROM python:3.11-slim AS production-stage

# Install system dependencies for ChromaDB and sqlite
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Ensure CrewAI and Langchain are installed completely
RUN pip install --no-cache-dir langchain-core langchain crewai

# Copy backend source code
COPY . .

# Copy built frontend assets from the build stage to the backend directory
COPY --from=build-stage /app/frontend/dist /app/src/web_app/frontend/dist

# Expose the port that FastAPI will run on
EXPOSE 8080

# Run the FastAPI server using Uvicorn
CMD ["uvicorn", "src.web_app.api:app", "--host", "0.0.0.0", "--port", "8080"]
