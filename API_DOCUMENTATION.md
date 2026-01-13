# RAG Document Search API

## Official API Documentation

**Version:** 1.0.0
**Base URL:** `https://your-api-domain.up.railway.app`

---

## Table of Contents

1. [Introduction](#introduction)
2. [Authentication](#authentication)
3. [Quick Start](#quick-start)
4. [API Endpoints](#api-endpoints)
   - [Health Check](#health-check)
   - [Query Documents](#query-documents)
   - [Ingest File](#ingest-file)
   - [Ingest Text](#ingest-text)
5. [Code Examples](#code-examples)
   - [PHP](#php)
   - [JavaScript / Node.js](#javascript--nodejs)
   - [Python](#python)
   - [cURL](#curl)
   - [C# / .NET](#c--net)
   - [Java](#java)
   - [Go](#go)
   - [Ruby](#ruby)
6. [Response Formats](#response-formats)
7. [Error Handling](#error-handling)
8. [Rate Limits & Best Practices](#rate-limits--best-practices)
9. [FAQ](#faq)
10. [Support](#support)

---

## Introduction

### What is This Service?

The **RAG Document Search API** is an intelligent document search and question-answering service powered by AI. Upload your documents once, then ask questions in natural language and receive accurate, context-aware answers.

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   1. UPLOAD DOCUMENTS                                           │
│      Upload PDFs, Word docs, or text files to the API           │
│      Documents are processed, chunked, and stored securely      │
│                                                                 │
│   2. ASK QUESTIONS                                              │
│      Send natural language questions to the API                 │
│      "What is our refund policy?"                               │
│      "How do I reset my password?"                              │
│                                                                 │
│   3. GET INTELLIGENT ANSWERS                                    │
│      Receive AI-generated answers based on YOUR documents       │
│      Answers include source references for verification         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Natural Language Queries** | Ask questions in plain English |
| **Multi-Format Support** | PDF, DOCX, TXT, HTML files supported |
| **Source Citations** | Every answer includes document sources |
| **Secure & Isolated** | Your data is isolated from other clients |
| **Fast Response** | Typical response time under 3 seconds |
| **REST API** | Simple HTTP API works with any language |

### Use Cases

- **Customer Support** - Instant answers from knowledge bases
- **Internal Documentation** - Search company policies and procedures
- **Legal & Compliance** - Query contracts and regulatory documents
- **Research** - Search through academic papers and reports
- **Product Documentation** - Help users find answers quickly

---

## Authentication

All API requests (except `/health`) require authentication using an API key.

### API Key Format

Your API key is a string starting with `sk-` provided by your administrator.

```
sk-your-unique-api-key
```

### Using Your API Key

Include your API key in the `Authorization` header of every request:

```
Authorization: Bearer sk-your-api-key
```

### Example Request with Authentication

```bash
curl -X POST "https://your-api-domain.up.railway.app/api/v1/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-api-key" \
  -d '{"question": "What is the return policy?"}'
```

### Security Best Practices

- Never expose your API key in client-side code (browsers)
- Store API keys in environment variables, not in source code
- Rotate your API key if you suspect it has been compromised
- Use HTTPS for all API requests (enforced by default)

---

## Quick Start

### Step 1: Verify Your Connection

```bash
curl https://your-api-domain.up.railway.app/health
```

Expected response:
```json
{
  "status": "healthy"
}
```

### Step 2: Upload a Document

```bash
curl -X POST "https://your-api-domain.up.railway.app/api/v1/ingest" \
  -H "Authorization: Bearer sk-your-api-key" \
  -F "file=@/path/to/your/document.pdf"
```

### Step 3: Ask a Question

```bash
curl -X POST "https://your-api-domain.up.railway.app/api/v1/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-api-key" \
  -d '{"question": "What does this document say about pricing?"}'
```

---

## API Endpoints

### Health Check

Check if the API service is running.

```
GET /health
```

**Authentication:** Not required

**Response:**
```json
{
  "status": "healthy",
  "service": "rag-api",
  "version": "0.1.0"
}
```

---

### Query Documents

Ask a question and get an AI-generated answer based on your uploaded documents.

```
POST /api/v1/query
```

**Authentication:** Required

**Headers:**
```
Content-Type: application/json
Authorization: Bearer sk-your-api-key
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | Yes | The question to ask |
| `top_k` | integer | No | Number of source documents to consider (default: 5) |

**Example Request:**
```json
{
  "question": "What are the shipping options?",
  "top_k": 5
}
```

**Example Response:**
```json
{
  "answer": "Based on the documentation, there are three shipping options available: 1) Standard Shipping (5-7 business days) - Free for orders over $50, 2) Express Shipping (2-3 business days) - $9.99, and 3) Overnight Shipping (next business day) - $19.99. All orders are shipped via trusted carriers and include tracking information.",
  "sources": [
    {
      "document": "shipping_policy.pdf",
      "page": 2,
      "relevance_score": 0.94
    },
    {
      "document": "faq.pdf",
      "page": 5,
      "relevance_score": 0.87
    }
  ],
  "metadata": {
    "documents_searched": 5,
    "processing_time_ms": 1240
  }
}
```

---

### Ingest File

Upload a document file to be processed and indexed for searching.

```
POST /api/v1/ingest
```

**Authentication:** Required

**Headers:**
```
Authorization: Bearer sk-your-api-key
Content-Type: multipart/form-data
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | The document file to upload |

**Supported File Types:**

| Format | Extension | Max Size |
|--------|-----------|----------|
| PDF | `.pdf` | 50 MB |
| Word Document | `.docx` | 50 MB |
| Plain Text | `.txt` | 10 MB |
| HTML | `.html` | 10 MB |

**Example Response:**
```json
{
  "success": true,
  "document_name": "company_handbook.pdf",
  "document_id": "doc_abc123xyz",
  "chunk_count": 47,
  "message": "Document successfully processed and indexed"
}
```

---

### Ingest Text

Directly ingest raw text content without uploading a file.

```
POST /api/v1/ingest/text
```

**Authentication:** Required

**Headers:**
```
Content-Type: application/json
Authorization: Bearer sk-your-api-key
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `content` | string | Yes | The text content to ingest |
| `source_name` | string | No | A name to identify this content |
| `metadata` | object | No | Additional metadata to store |

**Example Request:**
```json
{
  "content": "Our return policy allows customers to return items within 30 days of purchase. Items must be in original condition with tags attached. Refunds are processed within 5-7 business days.",
  "source_name": "return_policy",
  "metadata": {
    "category": "policies",
    "last_updated": "2024-01-15"
  }
}
```

**Example Response:**
```json
{
  "success": true,
  "source_name": "return_policy",
  "chunk_count": 1,
  "message": "Text content successfully processed and indexed"
}
```

---

## Code Examples

### PHP

#### Query Documents
```php
<?php

class RAGClient {
    private string $apiUrl;
    private string $apiKey;

    public function __construct(string $apiUrl, string $apiKey) {
        $this->apiUrl = rtrim($apiUrl, '/');
        $this->apiKey = $apiKey;
    }

    /**
     * Ask a question and get an AI-generated answer
     */
    public function query(string $question, int $topK = 5): array {
        $ch = curl_init($this->apiUrl . '/api/v1/query');

        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode([
                'question' => $question,
                'top_k' => $topK
            ]),
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'Authorization: Bearer ' . $this->apiKey
            ]
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        if ($httpCode !== 200) {
            throw new Exception("API request failed with status: $httpCode");
        }

        return json_decode($response, true);
    }

    /**
     * Upload a document file
     */
    public function ingestFile(string $filePath): array {
        if (!file_exists($filePath)) {
            throw new Exception("File not found: $filePath");
        }

        $ch = curl_init($this->apiUrl . '/api/v1/ingest');

        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => [
                'file' => new CURLFile($filePath)
            ],
            CURLOPT_HTTPHEADER => [
                'Authorization: Bearer ' . $this->apiKey
            ]
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        if ($httpCode !== 200) {
            throw new Exception("API request failed with status: $httpCode");
        }

        return json_decode($response, true);
    }

    /**
     * Ingest raw text content
     */
    public function ingestText(string $content, string $sourceName = null): array {
        $ch = curl_init($this->apiUrl . '/api/v1/ingest/text');

        $payload = ['content' => $content];
        if ($sourceName) {
            $payload['source_name'] = $sourceName;
        }

        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($payload),
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'Authorization: Bearer ' . $this->apiKey
            ]
        ]);

        $response = curl_exec($ch);
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        if ($httpCode !== 200) {
            throw new Exception("API request failed with status: $httpCode");
        }

        return json_decode($response, true);
    }
}

// Usage Example
$client = new RAGClient(
    'https://your-api-domain.up.railway.app',
    'sk-your-api-key'
);

// Upload a document
try {
    $result = $client->ingestFile('/path/to/document.pdf');
    echo "Uploaded: {$result['chunk_count']} chunks\n";
} catch (Exception $e) {
    echo "Error: " . $e->getMessage() . "\n";
}

// Ask a question
try {
    $result = $client->query("What is the refund policy?");
    echo "Answer: " . $result['answer'] . "\n";
    echo "Sources: " . json_encode($result['sources']) . "\n";
} catch (Exception $e) {
    echo "Error: " . $e->getMessage() . "\n";
}
```

---

### JavaScript / Node.js

#### Using Fetch (Browser & Node.js 18+)
```javascript
class RAGClient {
  constructor(apiUrl, apiKey) {
    this.apiUrl = apiUrl.replace(/\/$/, '');
    this.apiKey = apiKey;
  }

  /**
   * Ask a question and get an AI-generated answer
   * @param {string} question - The question to ask
   * @param {number} topK - Number of documents to consider (default: 5)
   * @returns {Promise<Object>} The query response
   */
  async query(question, topK = 5) {
    const response = await fetch(`${this.apiUrl}/api/v1/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify({ question, top_k: topK })
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Upload a document file
   * @param {File|Blob} file - The file to upload
   * @returns {Promise<Object>} The ingest response
   */
  async ingestFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.apiUrl}/api/v1/ingest`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: formData
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }

    return response.json();
  }

  /**
   * Ingest raw text content
   * @param {string} content - The text content to ingest
   * @param {string} sourceName - Optional name for the content
   * @returns {Promise<Object>} The ingest response
   */
  async ingestText(content, sourceName = null) {
    const payload = { content };
    if (sourceName) payload.source_name = sourceName;

    const response = await fetch(`${this.apiUrl}/api/v1/ingest/text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.apiKey}`
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.status}`);
    }

    return response.json();
  }
}

// Usage Example
const client = new RAGClient(
  'https://your-api-domain.up.railway.app',
  'sk-your-api-key'
);

// Ask a question
async function main() {
  try {
    const result = await client.query("What are the shipping options?");
    console.log("Answer:", result.answer);
    console.log("Sources:", result.sources);
  } catch (error) {
    console.error("Error:", error.message);
  }
}

main();
```

#### Using Axios (Node.js)
```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

const API_URL = 'https://your-api-domain.up.railway.app';
const API_KEY = 'sk-your-api-key';

const client = axios.create({
  baseURL: API_URL,
  headers: {
    'Authorization': `Bearer ${API_KEY}`
  }
});

// Query
async function query(question) {
  const response = await client.post('/api/v1/query', { question });
  return response.data;
}

// Upload file
async function ingestFile(filePath) {
  const form = new FormData();
  form.append('file', fs.createReadStream(filePath));

  const response = await client.post('/api/v1/ingest', form, {
    headers: form.getHeaders()
  });
  return response.data;
}

// Usage
query("What is the return policy?")
  .then(result => console.log(result.answer))
  .catch(err => console.error(err));
```

---

### Python

```python
import requests
from pathlib import Path
from typing import Optional, Dict, Any


class RAGClient:
    """Client for the RAG Document Search API."""

    def __init__(self, api_url: str, api_key: str):
        """
        Initialize the RAG client.

        Args:
            api_url: The base URL of the API
            api_key: Your API key
        """
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}'
        })

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Ask a question and get an AI-generated answer.

        Args:
            question: The question to ask
            top_k: Number of documents to consider (default: 5)

        Returns:
            Dictionary containing answer and sources
        """
        response = self.session.post(
            f'{self.api_url}/api/v1/query',
            json={'question': question, 'top_k': top_k},
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        return response.json()

    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """
        Upload and ingest a document file.

        Args:
            file_path: Path to the file to upload

        Returns:
            Dictionary containing ingest results
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, 'rb') as f:
            response = self.session.post(
                f'{self.api_url}/api/v1/ingest',
                files={'file': (path.name, f)}
            )
        response.raise_for_status()
        return response.json()

    def ingest_text(
        self,
        content: str,
        source_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ingest raw text content.

        Args:
            content: The text content to ingest
            source_name: Optional name for the content
            metadata: Optional metadata dictionary

        Returns:
            Dictionary containing ingest results
        """
        payload = {'content': content}
        if source_name:
            payload['source_name'] = source_name
        if metadata:
            payload['metadata'] = metadata

        response = self.session.post(
            f'{self.api_url}/api/v1/ingest/text',
            json=payload,
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy."""
        response = requests.get(f'{self.api_url}/health')
        response.raise_for_status()
        return response.json()


# Usage Example
if __name__ == '__main__':
    # Initialize client
    client = RAGClient(
        api_url='https://your-api-domain.up.railway.app',
        api_key='sk-your-api-key'
    )

    # Check health
    print("Health:", client.health_check())

    # Upload a document
    try:
        result = client.ingest_file('document.pdf')
        print(f"Uploaded: {result['chunk_count']} chunks")
    except FileNotFoundError as e:
        print(f"File error: {e}")
    except requests.HTTPError as e:
        print(f"API error: {e}")

    # Ask a question
    try:
        result = client.query("What is the refund policy?")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
    except requests.HTTPError as e:
        print(f"API error: {e}")
```

---

### cURL

#### Health Check
```bash
curl https://your-api-domain.up.railway.app/health
```

#### Query Documents
```bash
curl -X POST "https://your-api-domain.up.railway.app/api/v1/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-api-key" \
  -d '{
    "question": "What is the return policy?",
    "top_k": 5
  }'
```

#### Upload a File
```bash
curl -X POST "https://your-api-domain.up.railway.app/api/v1/ingest" \
  -H "Authorization: Bearer sk-your-api-key" \
  -F "file=@/path/to/document.pdf"
```

#### Ingest Text
```bash
curl -X POST "https://your-api-domain.up.railway.app/api/v1/ingest/text" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-api-key" \
  -d '{
    "content": "Your text content here...",
    "source_name": "my_content"
  }'
```

---

### C# / .NET

```csharp
using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

public class RAGClient : IDisposable
{
    private readonly HttpClient _client;
    private readonly string _apiUrl;

    public RAGClient(string apiUrl, string apiKey)
    {
        _apiUrl = apiUrl.TrimEnd('/');
        _client = new HttpClient();
        _client.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Bearer", apiKey);
    }

    public async Task<JsonDocument> QueryAsync(string question, int topK = 5)
    {
        var payload = new { question, top_k = topK };
        var content = new StringContent(
            JsonSerializer.Serialize(payload),
            Encoding.UTF8,
            "application/json"
        );

        var response = await _client.PostAsync($"{_apiUrl}/api/v1/query", content);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync();
        return JsonDocument.Parse(json);
    }

    public async Task<JsonDocument> IngestFileAsync(string filePath)
    {
        using var form = new MultipartFormDataContent();
        using var fileStream = File.OpenRead(filePath);
        using var fileContent = new StreamContent(fileStream);

        form.Add(fileContent, "file", Path.GetFileName(filePath));

        var response = await _client.PostAsync($"{_apiUrl}/api/v1/ingest", form);
        response.EnsureSuccessStatusCode();

        var json = await response.Content.ReadAsStringAsync();
        return JsonDocument.Parse(json);
    }

    public void Dispose() => _client.Dispose();
}

// Usage
class Program
{
    static async Task Main()
    {
        using var client = new RAGClient(
            "https://your-api-domain.up.railway.app",
            "sk-your-api-key"
        );

        var result = await client.QueryAsync("What is the return policy?");
        Console.WriteLine(result.RootElement.GetProperty("answer").GetString());
    }
}
```

---

### Java

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;

public class RAGClient {
    private final String apiUrl;
    private final String apiKey;
    private final HttpClient client;

    public RAGClient(String apiUrl, String apiKey) {
        this.apiUrl = apiUrl.replaceAll("/$", "");
        this.apiKey = apiKey;
        this.client = HttpClient.newHttpClient();
    }

    public String query(String question) throws Exception {
        String json = String.format("{\"question\": \"%s\"}", question);

        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(apiUrl + "/api/v1/query"))
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer " + apiKey)
            .POST(HttpRequest.BodyPublishers.ofString(json))
            .build();

        HttpResponse<String> response = client.send(
            request,
            HttpResponse.BodyHandlers.ofString()
        );

        if (response.statusCode() != 200) {
            throw new RuntimeException("API error: " + response.statusCode());
        }

        return response.body();
    }

    public static void main(String[] args) throws Exception {
        RAGClient client = new RAGClient(
            "https://your-api-domain.up.railway.app",
            "sk-your-api-key"
        );

        String result = client.query("What is the return policy?");
        System.out.println(result);
    }
}
```

---

### Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
)

type RAGClient struct {
    APIUrl string
    APIKey string
    Client *http.Client
}

type QueryRequest struct {
    Question string `json:"question"`
    TopK     int    `json:"top_k,omitempty"`
}

type QueryResponse struct {
    Answer  string                   `json:"answer"`
    Sources []map[string]interface{} `json:"sources"`
}

func NewRAGClient(apiUrl, apiKey string) *RAGClient {
    return &RAGClient{
        APIUrl: apiUrl,
        APIKey: apiKey,
        Client: &http.Client{},
    }
}

func (c *RAGClient) Query(question string) (*QueryResponse, error) {
    payload := QueryRequest{Question: question, TopK: 5}
    jsonData, _ := json.Marshal(payload)

    req, _ := http.NewRequest(
        "POST",
        c.APIUrl+"/api/v1/query",
        bytes.NewBuffer(jsonData),
    )
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Authorization", "Bearer "+c.APIKey)

    resp, err := c.Client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    body, _ := io.ReadAll(resp.Body)

    var result QueryResponse
    json.Unmarshal(body, &result)
    return &result, nil
}

func main() {
    client := NewRAGClient(
        "https://your-api-domain.up.railway.app",
        "sk-your-api-key",
    )

    result, err := client.Query("What is the return policy?")
    if err != nil {
        fmt.Println("Error:", err)
        return
    }

    fmt.Println("Answer:", result.Answer)
}
```

---

### Ruby

```ruby
require 'net/http'
require 'json'
require 'uri'

class RAGClient
  def initialize(api_url, api_key)
    @api_url = api_url.chomp('/')
    @api_key = api_key
  end

  def query(question, top_k: 5)
    uri = URI("#{@api_url}/api/v1/query")

    request = Net::HTTP::Post.new(uri)
    request['Content-Type'] = 'application/json'
    request['Authorization'] = "Bearer #{@api_key}"
    request.body = { question: question, top_k: top_k }.to_json

    response = Net::HTTP.start(uri.hostname, uri.port, use_ssl: true) do |http|
      http.request(request)
    end

    JSON.parse(response.body)
  end

  def ingest_file(file_path)
    uri = URI("#{@api_url}/api/v1/ingest")

    form_data = [['file', File.open(file_path)]]
    request = Net::HTTP::Post.new(uri)
    request['Authorization'] = "Bearer #{@api_key}"
    request.set_form(form_data, 'multipart/form-data')

    response = Net::HTTP.start(uri.hostname, uri.port, use_ssl: true) do |http|
      http.request(request)
    end

    JSON.parse(response.body)
  end
end

# Usage
client = RAGClient.new(
  'https://your-api-domain.up.railway.app',
  'sk-your-api-key'
)

result = client.query("What is the return policy?")
puts "Answer: #{result['answer']}"
```

---

## Response Formats

### Successful Query Response

```json
{
  "answer": "The AI-generated answer based on your documents...",
  "sources": [
    {
      "document": "filename.pdf",
      "page": 3,
      "relevance_score": 0.95
    }
  ],
  "metadata": {
    "documents_searched": 5,
    "processing_time_ms": 1240
  }
}
```

### Successful Ingest Response

```json
{
  "success": true,
  "document_name": "uploaded_file.pdf",
  "document_id": "doc_abc123",
  "chunk_count": 47,
  "message": "Document successfully processed and indexed"
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| `200` | Success | Request completed successfully |
| `400` | Bad Request | Invalid request format or missing required fields |
| `401` | Unauthorized | Missing or invalid API key |
| `403` | Forbidden | API key doesn't have permission |
| `404` | Not Found | Endpoint doesn't exist |
| `413` | Payload Too Large | File exceeds maximum size |
| `415` | Unsupported Media Type | File format not supported |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server-side error |

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_API_KEY",
    "message": "The provided API key is invalid or expired",
    "details": null
  }
}
```

### Common Errors

| Error Code | Cause | Solution |
|------------|-------|----------|
| `INVALID_API_KEY` | Wrong or expired API key | Check your API key |
| `MISSING_QUESTION` | No question provided | Include `question` in body |
| `UNSUPPORTED_FILE_TYPE` | File format not supported | Use PDF, DOCX, TXT, or HTML |
| `FILE_TOO_LARGE` | File exceeds 50MB limit | Split into smaller files |
| `NO_DOCUMENTS_FOUND` | No documents ingested yet | Upload documents first |

---

## Rate Limits & Best Practices

### Rate Limits

| Endpoint | Limit |
|----------|-------|
| `/api/v1/query` | 60 requests/minute |
| `/api/v1/ingest` | 10 requests/minute |
| `/api/v1/ingest/text` | 30 requests/minute |

### Best Practices

1. **Cache Responses**
   Cache query responses when appropriate to reduce API calls.

2. **Batch Uploads**
   Upload multiple documents sequentially, not in parallel.

3. **Handle Errors Gracefully**
   Implement retry logic with exponential backoff for 429/500 errors.

4. **Use Specific Questions**
   More specific questions yield better answers.
   - Bad: "Tell me about the product"
   - Good: "What are the dimensions and weight of the XL widget?"

5. **Keep Documents Updated**
   Re-ingest documents when their content changes.

6. **Monitor Usage**
   Track your API usage to stay within limits.

---

## FAQ

**Q: How long are documents stored?**
A: Documents are stored indefinitely until you delete them or your account is closed.

**Q: Can I delete specific documents?**
A: Contact support to request document deletion.

**Q: What languages are supported?**
A: The API works best with English documents but can handle most Latin-alphabet languages.

**Q: Is my data secure?**
A: Yes. All data is encrypted in transit (HTTPS) and at rest. Your documents are isolated from other clients.

**Q: How accurate are the answers?**
A: Answers are generated based on your uploaded documents. The AI cites sources so you can verify accuracy.

**Q: Can I use this in production?**
A: Yes, the API is designed for production use with high availability.

---

## Support

For technical support or questions:

- **Documentation**: You're reading it!
- **API Status**: Check `/health` endpoint
- **Interactive Docs**: Visit `/docs` on the API URL
- **Email Support**: Contact your administrator

---

*Last Updated: January 2026*
