# Diagrams

These use Mermaid syntax — viewable directly on GitHub, in VS Code
(with a Mermaid extension), or at https://mermaid.live by pasting the
code block contents.

## 1. Overall System Architecture

```mermaid
flowchart TD
    A[PDF Documents] --> B[Text Extraction - pypdf]
    B --> C[Text Chunking - LangChain Splitter]
    C --> D[Embedding Generation - Sentence-Transformers]
    D --> E[FAISS Vector Store]
    F[User Question] --> G[Question Embedding]
    G --> H[FAISS Similarity Search]
    E --> H
    H --> I[Top-K Relevant Chunks]
    I --> J[LLM - Groq API]
    J --> K[Generated Answer]
    K --> L[Source + Page Citation]
```

## 2. RAG Pipeline

```mermaid
flowchart LR
    Q[Question] --> R[Retrieve: FAISS top-k chunks]
    R --> AUG[Augment: build prompt with context + question]
    AUG --> G[Generate: LLM answers from context only]
    G --> OUT[Answer + Sources]
```

## 3. Data Flow

```mermaid
flowchart TD
    Upload[User uploads PDFs] --> Extract[Extract text per page]
    Extract --> Chunk[Split into overlapping chunks]
    Chunk --> Embed[Generate embeddings]
    Embed --> Index[Store in FAISS index]
    Index --> Session[Held in Streamlit session_state]
    Session --> Ask[User asks a question]
    Ask --> Search[Search FAISS for top-k chunks]
    Search --> LLM[Send question + chunks to LLM]
    LLM --> Display[Display answer, category, sources]
    Session --> Risk[Risk scan over all chunks]
    Risk --> Report[Summary report generation]
    Display --> Report
```

## 4. Use Case Diagram

```mermaid
flowchart LR
    User((Student / Analyst))
    User --> UC1[Upload Documents]
    User --> UC2[Ask Question]
    User --> UC3[View Answer + Sources]
    User --> UC4[Enter Financial Figures]
    User --> UC5[Calculate Ratios]
    User --> UC6[Scan for Risks]
    User --> UC7[Generate Summary Report]
    User --> UC8[Download Report]
```

## 5. Sequence Diagram — Asking a Question

```mermaid
sequenceDiagram
    participant U as User
    participant S as Streamlit UI
    participant E as Embeddings Module
    participant F as FAISS Index
    participant L as LLM (Groq)

    U->>S: Enter question, click Analyze
    S->>E: embed_texts([question])
    E-->>S: question embedding
    S->>F: search(index, question_embedding, top_k)
    F-->>S: top-k chunk indices + scores
    S->>S: resolve indices to chunk text
    S->>L: system prompt + context + question
    L-->>S: generated answer
    S-->>U: Answer, category, sources displayed
```
