# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --timeout=200

COPY . /app
ARG input_path
RUN mkdir -p /app/${input_path}


EXPOSE 8000
EXPOSE 8501

# Run FastAPI with Uvicorn as the ASGI server
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["streamlit", "run", "app.py"]
