# 📊 Plagiarism Detection Dashboard

An interactive web application built with Streamlit to analyze text documents for plagiarism. It leverages Natural Language Processing (NLP) models to calculate similarity scores and presents the results in a user-friendly dashboard with interactive charts and key statistics.

![Dashboard Screenshot](https://i.imgur.com/nJ6K8yv.png)

*(**Note to author:** You can replace the image link above with a fresh screenshot of your running application. Just drag and drop an image into the GitHub editor when editing this README to upload it and generate a new link.)*

---

## ✨ Features

-   **Multiple File Upload:** Drag and drop multiple `.txt` files for comparison.
-   **Dual Analysis Methods:**
    -   **Sentence-BERT (Advanced):** Uses deep learning to understand the semantic meaning of the text, making it excellent at detecting paraphrased content.
    -   **TF-IDF (Classic):** A traditional keyword-based statistical method, effective for finding direct copy-paste plagiarism.
-   **Interactive Dashboard:** A clean UI to visualize the analysis results.
-   **Key Metrics Summary:** At-a-glance view of the most important statistics, including total pairs compared, pairs above the threshold, highest similarity, and average similarity.
-   **Rich Visualizations:**
    -   Bar chart showing the most similar pairs of documents.
    -   Histogram displaying the distribution of similarity scores.
    -   Grouped bar chart for per-document analysis, showing the maximum and average similarity for each file.
-   **Downloadable Reports:** Export the full similarity report to a CSV file for further analysis or record-keeping.

---

## 🛠️ Technology Stack

-   **Backend:** Python
-   **Web Framework:** Streamlit
-   **NLP & Machine Learning:**
    -   `sentence-transformers` (Hugging Face) for the Sentence-BERT model.
    -   `scikit-learn` for TF-IDF and Cosine Similarity calculations.
-   **Data Manipulation:** Pandas & NumPy
-   **Data Visualization:** Plotly

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

-   Python 3.8 or newer
-   `pip` package manager

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/barshal-horse/Plagarism-detection-using-NLP.git
    ```

2.  **Navigate to the project directory:**
    ```bash
    cd Plagarism-detection-using-NLP
    ```

3.  **(Recommended) Create and activate a virtual environment:**
    -   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    -   **macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### How to Run the Application

1.  **Run the Streamlit app from your terminal:**
    ```bash
    streamlit run app.py
    ```
    *If the command above doesn't work due to PATH issues, use this alternative:*
    ```bash
    python -m streamlit run app.py
    ```

2.  **Open your web browser** and navigate to the local URL provided in the terminal (usually `http://localhost:8501`).

3.  Use the sidebar to upload your `.txt` files, select an analysis method, set the threshold, and click "Analyze Documents" to see the dashboard.

---
