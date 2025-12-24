# What's Good: AI News Analyst (Production Architecture)

A stateful, real-time RAG system that learns user preferences via vector arithmetic. Built on AWS microservices.

---

## The System: "Hello World" vs. Production

Most RAG tutorials are static: they retrieve the same 5 chunks every time. What's Good is dynamic. It implements a "Living Persona" engine that mathematically modifies the query vector based on every user interaction (Like/Dislike), delivering real-time personalization with <800ms latency.

---

## High-Level Architecture

**Hybrid Compute:**  
Uses ECS Fargate for stateful, memory-heavy inference (Recommendation Engine) and Serverless Lambda for stateless, bursty tasks (Ingestion/User Auth).

**Cost Optimization:**  
Decoupling the ingestion service reduced monthly cloud spend by ~60% (vs. always-on containers).

```
graph TD
    User(User) -->|HTTPS| API(AWS API Gateway)
    
    subgraph "Stateless (Serverless Lambda)"
        API -->|/auth| Auth[User Service<br/>(Python/Lambda)]
        API -->|/interact| Interact[Interaction Service<br/>(Python/Lambda)]
    end
    
    subgraph "Stateful (ECS Fargate)"
        API -->|/recommend| LB(Load Balancer)
        LB --> Engine[Recommendation Engine<br/>(FastAPI + In-Memory Models)]
    end
    
    subgraph "Data Pipeline"
        Cron(EventBridge) -->|Trigger| Ingest[Ingestion Service<br/>(Lambda + Docker)]
        Ingest -->|Write| VectorDB[(Pinecone)]
        Ingest -->|Write| SQL[(Supabase)]
    end

    Engine <--> VectorDB
    Engine <--> SQL
    Engine -->|Synthesis| LLM[Groq / Llama 3]
```
---

## Key Engineering Implementations

### 1. Performance Engineering: From 17s to 6s
I initially faced a massive latency spike (17s p95) due to **AWS Fargate CPU throttling** during sequential processing.
* **The Fix:** I refactored the pipeline to use **Asynchronous Fan-Out** (`asyncio.gather`), firing 5 parallel LLM generation requests instead of blocking the event loop sequentially.
* **The Result:** Shifted the system from CPU-bound to I/O-bound, stabilizing latency at **~6s** even under burst load.

### 2. Vector Arithmetic for Real-Time Learning

Instead of static retrieval, I implemented Vector Interpolation using NumPy. The system modifies the user's "Persona Vector" in real-time.

**Logic:**

New_Query = Base_Vector + (Weight * Liked_Vector) - (Weight * Disliked_Vector)

yaml
Copy code

**Result:**  
The feed adapts instantly to user sentiment without retraining the model.

---

### 3. The "Cold Start" Optimization (Lazy Loading)

**The Problem:**  
The sentence-transformers model is >400MB. Loading it on every Lambda invocation caused timeouts.

**The Solution:**  
Implemented a Lazy-Loading Singleton Pattern. The model loads once into the execution environment's memory and persists across warm invocations.

**Impact:**  
Reduced average latency from 10s (cold) to <200ms (warm).

---

### 4. CI/CD Cross-Compilation (ARM64 vs x86)

**The Problem:**  
Developing on a Mac M3 (ARM64) but deploying to AWS Lambda (x86_64) caused "Exec Format Errors" in production Docker containers.

**The Solution:**  
Engineered a custom GitHub Actions pipeline using QEMU emulation.

**Code:**  
`.github/workflows/deploy-ingestion-service.yml`

**Result:**  
Automated, reliable cross-architecture builds pushed directly to AWS ECR.

---

## Tech Stack & Metrics

**Infrastructure:**  
AWS (ECS Fargate, Lambda, API Gateway, EventBridge, ECR)

**ML Ops:**  
Docker, GitHub Actions (OIDC Security), Pinecone, Supabase

**AI:**  
LangChain, SentenceTransformers (Bi/Cross-Encoders), Groq (Llama 3)

---

## Performance

* **p95 Retrieval Latency:** <850ms (Vector Search + Re-Ranking)
* **End-to-End Latency:** ~6s (Includes Parallel LLM Generation)
* **Throughput:** Handles 15+ concurrent users via FastAPI Async Workers
