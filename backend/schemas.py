# schemas.py

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class CriterionCreate(BaseModel):
    name: str
    description: str


class CriterionOut(BaseModel):
    id: int
    name: str
    description: str

    class Config:
        from_attributes = True


class JobAnalysisCriterionCreate(BaseModel):
    criterion_id: int
    weight: float


class JobAnalysisCreate(BaseModel):
    prompt: str
    criteria: List[JobAnalysisCriterionCreate]


class JobAnalysisOut(BaseModel):
    id: int
    prompt: str
    created_at: datetime

    class Config:
        from_attributes = True


class CVScoreCreate(BaseModel):
    job_analysis_criterion_id: int
    score: float
    explanation: Optional[str]


class CVScoreOut(BaseModel):
    job_analysis_criterion_id: int
    score: float
    explanation: Optional[str]

    class Config:
        from_attributes = True


class CVAnalysisCreate(BaseModel):
    job_analysis_id: int
    filename: str
    candidate_name: str
    summary: Optional[str]
    total_score: Optional[float]
    scores: List[CVScoreCreate]


class CVAnalysisOut(BaseModel):
    id: int
    filename: str
    candidate_name: str
    summary: Optional[str]
    total_score: Optional[float]
    scores: List[CVScoreOut]

    class Config:
        from_attributes = True

