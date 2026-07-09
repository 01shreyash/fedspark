# FedSpark — Streaming-Aware, Byzantine-Robust Federated Learning on Apache Spark

A cross-silo federated learning system where each client is itself a distributed Apache Spark application, designed for financial fraud detection. Banks with data too large for any single machine can jointly train a fraud-detection model — with streaming ingestion, non-IID-robust + poisoning-robust aggregation, and differential privacy.

## Architecture

```
Generator (PaySim replay) → Kafka/Redpanda → Silo (Spark Streaming → Lakehouse → Trainer) → Coordinator (QWRA Aggregator) → Dashboard
```

## Quick Start

### Prerequisites

- Docker Desktop (WSL2 backend on Windows)
- 8 GB+ RAM (16 GB recommended)
- Python 3.11 (for local dev/testing)

### Run

```bash
cd fedspark

# Start all services
make dev-up

# Check health
curl http://localhost:8000/healthz

# View dashboard
open http://localhost:8501

# Stop everything
make clean
```

### Run Tests

```bash
make test
```

### Run Experiments

```bash
make experiment E=E1   # FL vs centralized utility
make experiment E=E2   # Non-IID robustness
make experiment E=E3   # Poisoning robustness
make experiment E=E4   # Drift adaptation
make experiment E=E5   # Scalability
make experiment E=E6   # Privacy-utility frontier
```

## Project Structure

```
fedspark/
├── docker-compose.yml       # Multi-container orchestration
├── Dockerfile               # Python 3.11-slim base
├── Makefile                 # Dev/test/experiment targets
├── configs/
│   ├── base.yaml            # Default configuration
│   └── experiments/         # E1-E6 experiment configs
├── common/
│   ├── model.py             # NumPy MLP (forward/backward/serialize)
│   ├── serialize.py         # Base64 .npz encoding
│   ├── schemas.py           # Pydantic API schemas
│   └── metrics.py           # SQLite metrics store
├── generator/
│   ├── amplify.py           # PaySim data amplification
│   ├── partition.py         # Non-IID Dirichlet partitioning
│   ├── replay.py            # Kafka replay
│   └── inject.py            # Label flip / drift / poisoning injection
├── silo/
│   ├── client.py            # FL round loop (poll → train → submit)
│   ├── trainer.py           # Mode A/B distributed training
│   ├── features.py          # Spark SQL feature pipeline
│   ├── ingest.py            # Structured Streaming from Kafka
│   └── drift.py             # PSI concept-drift monitor
├── coordinator/
│   ├── api.py               # FastAPI control plane
│   ├── aggregate.py         # FedAvg / QWRA / Trimmed Mean
│   ├── registry.py          # Versioned model store
│   └── privacy.py           # Analytic Gaussian DP accounting
├── dashboard/app.py         # Streamlit live dashboard
├── experiments/
│   ├── runner.py            # Config-driven experiment harness
│   └── plots.py             # CSV → figures → RESULTS.md
└── tests/                   # pytest unit + integration tests
```

## APIs

| Method | Path | Purpose |
|--------|------|---------|
| POST | /register | Silo enrollment |
| GET | /round/current | Poll for work |
| GET | /model/{version} | Fetch global model |
| POST | /update | Submit local update |
| POST | /drift | Drift signal |
| GET | /metrics/latest | Dashboard data |
| GET | /healthz | Health check |

## Key Algorithms

- **Quality-Weighted Robust Aggregation (QWRA)**: Combines data size, cosine-similarity outlier rejection, and quality scoring on a public validation set
- **FedProx**: Proximal term stabilizes non-IID training
- **Semi-synchronous rounds**: Straggler-tolerant with bounded staleness
- **DP via analytic Gaussian mechanism**: Per-round ε accounting

## Experiments

| ID | Question | Expected Result |
|----|----------|----------------|
| E1 | FL vs centralized utility | FedSpark within ~5% of centralized AUC-PR |
| E2 | Non-IID robustness | QWRA degrades least at α=0.1 |
| E3 | Poisoning robustness | QWRA holds within a few points of clean run |
| E4 | Drift adaptation | FedSpark recovers within a few rounds of PSI alert |
| E5 | Scalability | Spark client scales near-linearly; pandas OOMs at 25M |
| E6 | Privacy-utility frontier | Usable utility at single-digit-to-moderate ε |

## Tech Stack

| Layer | Choice |
|-------|--------|
| Runtime | Python 3.11 (Linux containers) |
| Compute | PySpark 3.5 |
| Lakehouse | Delta/Parquet |
| Stream bus | Redpanda (Kafka API) |
| Control plane | FastAPI + uvicorn |
| Model | NumPy MLP (no torch/tf) |
| Dashboard | Streamlit + Plotly |
| Orchestration | Docker Compose |

## License

MIT — for educational purposes.
