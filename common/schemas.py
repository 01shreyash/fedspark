from pydantic import BaseModel


class RegisterRequest(BaseModel):
    silo_id: str
    n_rows: int


class RegisterResponse(BaseModel):
    token: str


class RoundResponse(BaseModel):
    round: int | None = None
    version: int | None = None
    deadline_utc: str | None = None
    hyperparams: dict | None = None
    idle: bool = False


class UpdateRequest(BaseModel):
    silo_id: str
    round: int
    delta_b64: str
    n_samples: int
    train_loss: float
    wall_time_s: float
    token: str = ""


class UpdateResponse(BaseModel):
    accepted: bool
    staleness: int = 0


class DriftRequest(BaseModel):
    silo_id: str
    psi: float
    feature: str = "amount"


class ModelResponse(BaseModel):
    version: int
    weights_b64: str
    arch: dict
