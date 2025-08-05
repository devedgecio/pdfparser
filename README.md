# PDF Parser App - README

Welcome to the **PDF Parser App**! This guide will help you set up and run the app in three different ways: using **Streamlit**, **FastAPI**, and **Docker**. Follow the instructions based on the method you'd like to use.


---

## 1. Running the App with Docker

### Prerequisites:
- **Docker** must be installed on your machine.
- **Docker Compose**  must be installed on your machine.
- **Python version 3.10** must be installed on your machine.

### Steps to Run the App with Docker:

1. **Navigate to the Project Directory**:

2. **Set up Enviroment Variables**

    ```bash
    cp ./.env_example ./.env
    ```

    **Note**
    Open .env file and add the API Keys
    you might need to enable hidden files view, as on some operating systems the files starting from **.** (dot) are ususally hidden.

2. **Paths Update**:
    - To give the input from the required directory, you have to get absolute path of the input directoy and set it in the .env and config.json
    - To get the output in the required directory, you have to get absolute path of the output directoy and set it in the .env and config.json
3. **Build the Docker Image**:
    To build the Docker image for the FastAPI app, use the following commands:
    ```bash
    sudo docker compose  up --build
    ```
4. **Access the App**:
    - **Streamlit**: Open your browser and go to `http://localhost:8501` to access the Sreamlit App.

---

## 2. Running the Streamlit App locally

### Prerequisites:
- Ensure you have **Python** and **Streamlit** installed on your machine.
- Install the required libraries using `requirements.txt`.

### Steps to Run the Streamlit App:

1. **Navigate to the Project Directory**:
    Open your terminal and navigate to the directory where the project files are located. Use the following command:
    ```bash
    cd ~/Path-to-PDF-PARSER-DIR
    ```
2. **Create Python Virtual Environment**:
    Run the following command to install the required dependencies:
    ```bash
    python3.10 -m venv .venv
    ```
    ```bash
    source .venv/bin/activate
    ```
    You should see something like (.venv) just at the start of directory path in the terminal
3. **Install Dependencies**:
    Run the following command to install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. **Paths Update**:
    - To give the input from the required directory, you have to get absolute path of the input directoy and set it in the config.json
    
    - To get the output in the required directory, you have to get absolute path of the output directoy and set it in the config.json
4. **Start the Streamlit App**:
    To run the Streamlit app, use the following command:
    ```bash
    streamlit run app.py
    ```

5. **Access the App**:
    Open your browser and go to `http://localhost:8501` to view the app.
    - The app will display two tables: **metadata** and a **final Table**.

6. **Steps to proceed**:
    - click on the Browse file button
    - pick a file that you want to process
    - just select and wait for processing to finish
    - after processing, you will see two tables **(1)** Metadata **(2)** Table
        

---

## 3. Running the FastAPI App

### Prerequisites:
- Ensure you have **Python** and **Uvicorn** installed on your machine.
- Install the required libraries using `requirements.txt`.

### Steps to Run the FastAPI EndPoints:

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
3. **Paths Update**:
    - To give the input from the required directory, you have to get absolute path of the input directoy and set it in the config.json
    
    - To get the output in the required directory, you have to get absolute path of the output directoy and set it in the config.json
4. **Start the FastAPI App**:
    To run the FastAPI app, use the following command:
    ```bash
    uvicorn api:app --reload
    ```

5. **Access the API**:
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
## 4. Running the FastAPI App
    Follow the steps:
    - Rename the .env_example to .env
    - Just set the values with your APIs and other required things
## Troubleshooting

1. **Port Conflicts**:
    If ports `8000` and `8501` are already in use, you can change the port numbers in the Docker commands or when running the apps.

2. **Missing Dependencies**:
    If you encounter errors related to missing dependencies, make sure you have installed all required packages with `pip install -r requirements.txt`.

---

## Conclusion

With these steps, you should be able to run the **PDF Parser App** in three different ways: via **Streamlit**, **FastAPI**, or **Docker**. If you face any issues, feel free to consult this README or contact the project maintainers for support.
