# PDF Parser App - README

Welcome to the **PDF Parser App**! This guide will help you set up and run the app in three different ways: using **Streamlit**, **FastAPI**, and **Docker**. Follow the instructions based on the method you'd like to use.

---

## 1. Running the Streamlit App

### Prerequisites:
- Ensure you have **Python** and **Streamlit** installed on your machine.
- Install the required libraries using `requirements.txt`.

### Steps to Run the Streamlit App:

1. **Navigate to the Project Directory**:
    Open your terminal and navigate to the directory where the project files are located. Use the following command:
    ```bash
    cd ~/Path-to-PDF-PARSER-DIR
    ```

2. **Install Dependencies**:
    Run the following command to install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. **Start the Streamlit App**:
    To run the Streamlit app, use the following command:
    ```bash
    streamlit run app.py
    ```

4. **Access the App**:
    Open your browser and go to `http://localhost:8501` to view the app.
    - The app will display two tables: **metadata** and a **final Table**.

---

## 2. Running the FastAPI App

### Prerequisites:
- Ensure you have **Python** and **Uvicorn** installed on your machine.
- Install the required libraries using `requirements.txt`.

### Steps to Run the FastAPI App:

1. **Navigate to the Project Directory**:
    Open your terminal and navigate to the project directory:
    ```bash
    cd ~/Path-to-PDF-PARSER-DIR
    ```

2. **Install Dependencies**:
    Run the following command to install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. **Start the FastAPI App**:
    To run the FastAPI app, use the following command:
    ```bash
    uvicorn api:app --reload
    ```

4. **Access the API**:
    Open your browser and go to `http://localhost:8000/docs` to access the FastAPI interactive documentation.
    - You can upload a ZIP file containing PDFs or just a single PDF.
    - Wait for the processing to finish.
    - At the bottom of the response section, you'll see a **JSON object** showing the status of each file, including whether it has all rows or not.
    - Download the processed ZIP by copying the `/tmp/tmp***.zip` URL and pasting it into a new browser tab.

    Inside the downloaded ZIP, you will find:
    - A folder for each file in the ZIP.
    - Inside each folder, you will find two files:
        - `filename.csv`: The table data.
        - `filename.json`: The metadata.

---

## 3. Running the App with Docker

### Prerequisites:
- **Docker** must be installed on your machine.

### Steps to Run the App with Docker:

1. **Navigate to the Project Directory**:
    Open your terminal and navigate to the project directory:
    ```bash
    cd ~/Path-to-PDF-PARSER-DIR
    ```

2. **Build the Docker Image**:
    To build the Docker image for the FastAPI or Streamlit app, use the following commands:
    
    For the **Streamlit app**:
    ```bash
    docker build -t pdf-parser-streamlit -f Dockerfile.streamlit .
    ```

    For the **FastAPI app**:
    ```bash
    docker build -t pdf-parser-fastapi -f Dockerfile.fastapi .
    ```

3. **Run the Docker Containers**:
    To run the app, use the following command:
    ```bash
    sudo docker run -d -p 8000:8000 -p 8501:8501 pdf-parser-fastapi
    ```

4. **Access the App**:
    - **Streamlit**: Open your browser and go to `http://localhost:8501` to access the Streamlit app.
    - **FastAPI**: Open your browser and go to `http://localhost:8000/docs` to access the FastAPI API documentation.

---

## 4. Environment Variables

You need to set the following **environment variables** in your `.env` file or directly in your terminal for the app to work correctly.

- `AZURE_FORM_KEY`: Placeholder for your Azure form key.
- `AZURE_FORM_ENDPOINT`: Example: `https://di-rextag1.cognitiveservices.azure.com/`
- `OPENAI_API_KEY`: Placeholder for your OpenAI API key.
- `OPENAI_MODEL`: Example: `gpt-4`
- `OPENAI_TIMEOUT`: Timeout duration for API requests (default: 15).
- `OPENAI_MAX_RETRIES`: Maximum retries for the API (default: 5).
- `AZURE_OPENAI_ENDPOINT`: Example: `https://hal1.openai.azure.com/`
- `AZURE_OPENAI_API_KEY`: Placeholder for your Azure OpenAI API key.
- `AZURE_OPENAI_DEPLOYMENT`: Example: `HAL1-4`
- `MODEL_NAME`: Example: `gpt-4.1-mini`

---

## Troubleshooting

1. **Port Conflicts**:
    If ports `8000` and `8501` are already in use, you can change the port numbers in the Docker commands or when running the apps.

2. **Missing Dependencies**:
    If you encounter errors related to missing dependencies, make sure you have installed all required packages with `pip install -r requirements.txt`.

---

## Conclusion

With these steps, you should be able to run the **PDF Parser App** in three different ways: via **Streamlit**, **FastAPI**, or **Docker**. If you face any issues, feel free to consult this README or contact the project maintainers for support.
