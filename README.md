# HakimAI - Arabic Medical Chatbot

Hakim is a specialized Artificial Intelligence API designed to answer medical, health, and biology questions in Arabic. It uses Retrieval-Augmented Generation (RAG) to fetch reliable information from trusted medical documents and provides accurate answers backed by source citations.

## What is it and how was it built?

Hakim was built to bridge the gap in reliable Arabic medical AI. Instead of guessing answers, it reads from a curated database of medical texts before responding. 

Here is the step-by-step process of how it was built:

1. **Data Collection:** A custom Python scraper systematically downloaded trusted Arabic medical PDFs from sources like MedlinePlus.
2. **Data Processing:** The raw text was cleaned to remove formatting errors and split into small, overlapping chunks to ensure the AI reads them without losing context.
3. **Vector Database:** Qdrant is used to store these text chunks. To make the text searchable, the BAAI/bge-m3 embedding model is used, allowing the system to perform hybrid searches (combining exact keyword matches with semantic meaning) using Reciprocal Rank Fusion (RRF).
4. **Language Model Integration:** The core brain is a fine-tuned model called Shams03/tawkeed-egy-medical-4b. A strict system prompt ensures the model only answers medical questions, refuses non-medical topics, and explicitly cites its sources.
5. **API & Containerization:** Everything is wrapped in a FastAPI application and containerized using Docker so it can be easily deployed on AWS with native GPU support.

## Prerequisites

Because the local language models and embeddings require heavy processing, your deployment environment must have:
* Docker and Docker Compose installed.
* An NVIDIA GPU.
* The NVIDIA Container Toolkit installed (required for the Docker container to access the GPU).

## How to Install and Run Locally

To get the application running on your own machine or server, run the following commands in your terminal:

```bash
git clone [https://github.com/mostafashams5/arabic-medical-chatbot.git](https://github.com/mostafashams5/arabic-medical-chatbot.git)
cd arabic-medical-chatbot/Arabic-Medical-Chatbot-be18734d5da98a9f2f3f365eecb3d6ebf7411a67
sudo docker compose up -d
sudo docker logs -f qwen_medical_api
```

The system will download the necessary models and start the Qdrant database and the FastAPI server. Once you see "AI Engine Online and Ready" in the logs, the system is good to go.

## API Integration Guide for Backend Developers

Hakim is exposed as a REST API. If you are integrating this model into a web dashboard, mobile app, or another backend system, here is everything you need to know.

**Endpoint URL:** `http://<YOUR-PUBLIC-IP>:8000/api/chat`
**Method:** `POST`
**Headers:** `Content-Type: application/json`

### Request Payload

You only need to send the user's question in the request body.

```json
{
    "question": "ما هي أسباب الصداع النصفي؟"
}
```

### Response Format

The server will return a JSON object containing the generated answer, the source document it pulled from, the page number, a confidence score, and the exact database chunks it read to form the answer.

```json
{
    "question": "ما هي أسباب الصداع النصفي؟",
    "answer": "بناءً على MedlinePlus - Headaches_ARA، صفحة 12: الصداع النصفي ينتج عن تغيرات في الدماغ...",
    "source": "MedlinePlus - Headaches_ARA",
    "page": "12",
    "score": 0.85,
    "retrieved_chunks": [
        {
            "source": "MedlinePlus - Headaches_ARA",
            "page": "12",
            "score": 0.85,
            "text": "النص الطبي المسترجع هنا..."
        }
    ]
}
```

### Python Example Code

Here is a ready-to-use Python snippet using the `requests` library to test the API. Make sure to replace the IP address with your actual AWS public IP or `localhost` if running locally.

```python
import requests

url = "[http://100.48.16.123:8000/api/chat](http://100.48.16.123:8000/api/chat)"

payload = {
    "question": "ما هي أسباب الصداع النصفي؟"
}

headers = {
    "Content-Type": "application/json"
}

print("Sending question to the server...")

try:
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print("\n--- Response Received ---")
        print("Question: ", data.get("question"))
        print("\nAnswer: ", data.get("answer"))
        print("\nSource: ", data.get("source"))
        print("Page: ", data.get("page"))
        print("Confidence Score: ", data.get("score"))
    else:
        print("Error. Status Code:", response.status_code)
        print("Details:", response.text)
except Exception as e:
    print("An error occurred:", str(e))
```
