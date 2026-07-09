import asyncio
import base64
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from common.metrics import init_db
from common.model import MLP
from common.schemas import (
    DriftRequest,
    ModelResponse,
    RegisterRequest,
    RegisterResponse,
    RoundResponse,
    UpdateRequest,
    UpdateResponse,
)
from common.serialize import weights_from_b64, weights_to_b64
from coordinator.aggregate import fedavg, qwra, trimmed_mean
from coordinator.privacy import compute_round_epsilon, compose_epsilons
from coordinator.registry import ModelRegistry

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(run_round_loop())
    yield
    task.cancel()

app = FastAPI(title="FedSpark Coordinator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

import os

config_path = Path("/app/configs/base.yaml")
if not config_path.exists():
    config_path = Path("configs/base.yaml")
with open(config_path) as f:
    CONFIG = yaml.safe_load(f)

_registry_path = os.environ.get("FEDSPARK_REGISTRY_PATH", "/app/registry")
_metrics_path = os.environ.get("FEDSPARK_METRICS_PATH", "/app/metrics/metrics.sqlite")
Path(_registry_path).mkdir(parents=True, exist_ok=True)
Path(_metrics_path).parent.mkdir(parents=True, exist_ok=True)

registry = ModelRegistry(registry_path=_registry_path)
metrics_db = init_db(_metrics_path)

run_config = CONFIG["run"]
data_config = CONFIG["data"]
model_config = CONFIG["model"]
train_config = CONFIG["train"]
protocol_config = CONFIG["protocol"]
aggregator_config = CONFIG["aggregator"]
privacy_config = CONFIG["privacy"]

N_FEATURES = data_config.get("features", model_config.get("features", 30))
if N_FEATURES == "auto":
    N_FEATURES = 30
HIDDEN = model_config.get("hidden", [64, 32])
SEED = run_config.get("seed", 42)
N_SILOS = data_config.get("silos", 3)
QUORUM = max(1, int(protocol_config.get("quorum_frac", 0.6) * N_SILOS))

global_model = MLP(n_features=N_FEATURES, hidden=HIDDEN, seed=SEED)
global_weights = {k: v.copy() for k, v in global_model.weights.items()}
current_round = 0
current_version = 0
round_deadline: float | None = None
round_active = False
silos_registered: dict[str, dict] = {}
silo_updates: dict[int, list[dict]] = {}
late_updates: list[dict] = []
drift_signals: dict[str, float] = {}
per_silo_epsilons: dict[str, list[float]] = {}
val_set: tuple[np.ndarray, np.ndarray] | None = None

init_weights_bytes = global_model.serialize()
init_version = registry.save(init_weights_bytes, {"round": 0, "type": "init"})
current_version = init_version


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/register", response_model=RegisterResponse)
async def register(req: RegisterRequest):
    if req.silo_id in silos_registered:
        raise HTTPException(400, "Silo already registered")
    token = f"tok_{req.silo_id}_{int(time.time())}"
    silos_registered[req.silo_id] = {"n_rows": req.n_rows, "token": token, "registered_at": time.time()}
    per_silo_epsilons[req.silo_id] = []
    return RegisterResponse(token=token)


@app.get("/round/current", response_model=RoundResponse)
async def round_current():
    global round_active
    if not round_active or current_round == 0:
        return RoundResponse(idle=True)
    return RoundResponse(
        round=current_round,
        version=current_version,
        deadline_utc=datetime.fromtimestamp(round_deadline, tz=timezone.utc).isoformat(),
        hyperparams={
            "lr": train_config.get("lr", 0.05),
            "local_epochs": train_config.get("local_epochs", 2),
            "prox_mu": train_config.get("prox_mu", 0.01),
            "batch_size": train_config.get("batch_size", 4096),
        },
    )


@app.get("/model/{version}", response_model=ModelResponse)
async def get_model(version: int):
    data = registry.load(version)
    if data is None:
        raise HTTPException(404, "Model version not found")
    b64 = base64.b64encode(data).decode("utf-8") if isinstance(data, bytes) else data
    return ModelResponse(version=version, weights_b64=b64, arch={"n_features": N_FEATURES, "hidden": HIDDEN})


@app.post("/update", response_model=UpdateResponse)
async def submit_update(req: UpdateRequest):
    if req.silo_id not in silos_registered:
        raise HTTPException(401, "Silo not registered")
    if req.round != current_round:
        return UpdateResponse(accepted=False, staleness=current_round - req.round if req.round < current_round else 0)
    update_data = {
        "silo_id": req.silo_id,
        "delta": weights_from_b64(req.delta_b64),
        "n_samples": req.n_samples,
        "train_loss": req.train_loss,
        "wall_time_s": req.wall_time_s,
        "arrived_at": time.time(),
    }
    if req.round not in silo_updates:
        silo_updates[req.round] = []
    silo_updates[req.round].append(update_data)
    return UpdateResponse(accepted=True, staleness=0)


@app.post("/drift")
async def report_drift(req: DriftRequest):
    drift_signals[req.silo_id] = req.psi
    metrics_db.execute(
        "INSERT INTO drift_alerts (round, silo_id, psi, feature) VALUES (?, ?, ?, ?)",
        (current_round, req.silo_id, req.psi, req.feature),
    )
    metrics_db.commit()
    return {"status": "logged"}


@app.get("/metrics/latest")
async def metrics_latest():
    cursor = metrics_db.execute("SELECT * FROM rounds ORDER BY round DESC LIMIT 10")
    rounds_data = [dict(row) for row in cursor.fetchall()]
    cursor = metrics_db.execute("SELECT * FROM silo_metrics ORDER BY id DESC LIMIT 50")
    silo_data = [dict(row) for row in cursor.fetchall()]
    return {"rounds": rounds_data, "silos": silo_data, "drift": drift_signals}


async def run_round_loop():
    global current_round, current_version, global_weights, round_deadline, round_active, silo_updates, late_updates
    n_rounds = protocol_config.get("rounds", 30)
    deadline_s = protocol_config.get("round_deadline_s", 120)
    quorum = QUORUM
    agg_kind = aggregator_config.get("kind", "qwra")
    server_lr = aggregator_config.get("server_lr", 1.0)
    size_exp = aggregator_config.get("size_exp", 0.5)
    cos_tau = aggregator_config.get("cos_tau", 0.0)
    staleness_lambda = protocol_config.get("staleness_lambda", 0.5)
    await asyncio.sleep(5)
    for r in range(1, n_rounds + 1):
        current_round = r
        round_deadline = time.time() + deadline_s
        round_active = True
        silo_updates[r] = []
        silo_updates[r] = silo_updates.get(r, [])
        deadline_extended = False
        while True:
            remaining = round_deadline - time.time()
            if remaining <= 0:
                break
            await asyncio.sleep(min(5, remaining))
            n_updates = len(silo_updates.get(r, []))
            if n_updates >= quorum:
                break
            if remaining < deadline_s / 2 and not deadline_extended and n_updates < quorum:
                round_deadline = time.time() + deadline_s / 2
                deadline_extended = True
        round_active = False
        updates = silo_updates.get(r, [])
        if late_updates:
            for lu in late_updates:
                lu["delta"] = {k: v * staleness_lambda for k, v in lu["delta"].items()}
                updates.append(lu)
            late_updates = []
        if len(updates) < quorum:
            late_updates = []
            continue
        deltas = [u["delta"] for u in updates]
        sizes = [silos_registered.get(u["silo_id"], {}).get("n_rows", 1) for u in updates]
        if agg_kind == "fedavg":
            agg_delta = fedavg(deltas, sizes, server_lr, size_exp)
        elif agg_kind == "trimmed_mean":
            agg_delta = trimmed_mean(deltas, trim_frac=0.2, server_lr=server_lr)
        else:
            agg_delta, rejected = qwra(
                deltas, sizes, global_weights, val_set, global_model,
                cos_tau=cos_tau, size_exp=size_exp, server_lr=server_lr,
            )
        for k in global_weights:
            global_weights[k] = global_weights[k] + agg_delta[k]
        global_model.weights = {k: v.copy() for k, v in global_weights.items()}
        weights_bytes = global_model.serialize()
        meta = {
            "round": r,
            "n_updates": len(updates),
            "agg_kind": agg_kind,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        new_version = registry.save(weights_bytes, meta)
        current_version = new_version
        metrics_db.execute(
            "INSERT INTO rounds (round, version, n_updates, wall_time_s) VALUES (?, ?, ?, ?)",
            (r, new_version, len(updates), deadline_s),
        )
        for u in updates:
            sid = u["silo_id"]
            flagged = 0
            if agg_kind == "qwra":
                u_idx = updates.index(u)
                if u_idx < len(rejected) and rejected[u_idx]:
                    flagged = 1
            eps = 0.0
            if privacy_config.get("enabled", False):
                eps = compute_round_epsilon(
                    privacy_config.get("sigma", 0.8),
                    privacy_config.get("delta", 1e-5),
                    privacy_config.get("clip", 1.0),
                    u["n_samples"],
                )
                per_silo_epsilons.setdefault(sid, []).append(eps)
            metrics_db.execute(
                "INSERT INTO silo_metrics (round, silo_id, n_samples, train_loss, flagged, epsilon) VALUES (?, ?, ?, ?, ?, ?)",
                (r, sid, u["n_samples"], u["train_loss"], flagged, eps),
            )
        metrics_db.commit()
        late_updates = []
        for u in updates:
            if u["arrived_at"] > round_deadline:
                late_updates.append(u)



