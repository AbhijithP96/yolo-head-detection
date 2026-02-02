FROM python:3.11

# Set working directory
WORKDIR /app

# Copy requirements file
COPY docker_req.txt /app/

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir -r docker_req.txt

# dagashub authentication setup
ARG DAGSHUB_ACCESS_TOKEN
ARG DAGSHUB_REPO_OWNER
ARG DAGSHUB_REPO_NAME

ENV DAGSHUB_ACCESS_TOKEN=${DAGSHUB_ACCESS_TOKEN}
ENV DAGSHUB_REPO_OWNER=${DAGSHUB_REPO_OWNER}
ENV DAGSHUB_REPO_NAME=${DAGSHUB_REPO_NAME}

# Copy the rest of the application code
COPY params.yaml /app/
COPY yolo_head_detection/config.py /app/yolo_head_detection/
COPY app.py /app/
COPY utils.py /app/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]