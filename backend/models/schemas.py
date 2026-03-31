# =====================================================
# models/schemas.py
# Pydantic models = the "shape" of your API.
#
# WHY Pydantic?
# - Automatic validation: if a field is wrong type,
#   FastAPI returns a 422 error with a clear message
# - Auto-generates API docs (Swagger UI)
# - Prevents you from accidentally returning passwords
#   (just don't include them in the response model)
# =====================================================

from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
from datetime import datetime
from enum import Enum


# ---- Enums (mirror the DB role_type enum) ----

class RoleType(str, Enum):
    """
    Mirror of the PostgreSQL role_type enum.
    Using str + Enum means it serializes as a string in JSON.
    """
    intern   = "intern"
    employee = "employee"
    manager  = "manager"
    admin    = "admin"


# ---- Auth Schemas ----

class SignUpRequest(BaseModel):
    """What the client sends to create a new account."""
    email: EmailStr           # Pydantic validates email format
    password: str
    department: str = "general"

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        """Enforce minimum password security."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    """What the client sends to log in."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """
    What we return after successful login.
    The access_token is the JWT — frontend stores this
    and sends it with every subsequent request.
    """
    access_token: str
    token_type: str = "bearer"
    expires_in: int            # seconds until token expires


class UserProfile(BaseModel):
    """
    Safe user profile — excludes sensitive fields.
    This is what /auth/me returns.
    """
    id: str
    email: str
    role: RoleType
    department: str
    is_active: bool
    created_at: datetime

    # Tell Pydantic to read from ORM objects (Supabase returns dicts)
    model_config = {"from_attributes": True}


# ---- Document Schemas ----

class DocumentMetadata(BaseModel):
    """
    Metadata attached when uploading a document.
    The uploader declares who should have access.
    NOTE: The backend validates this against the uploader's
    own role — you can't grant access higher than your own role.
    """
    role_access: list[RoleType]
    department: str = "general"

    @field_validator("role_access")
    @classmethod
    def at_least_one_role(cls, v: list) -> list:
        if not v:
            raise ValueError("role_access must contain at least one role")
        return v


class DocumentResponse(BaseModel):
    """What we return after a successful upload."""
    id: str
    filename: str
    file_type: str
    role_access: list[RoleType]
    department: str
    status: str
    chunk_count: int
    created_at: datetime


class DocumentListResponse(BaseModel):
    """Paginated list of documents the user can see."""
    documents: list[DocumentResponse]
    total: int


# ---- Query Schemas (used in Phase 5) ----

class QueryRequest(BaseModel):
    """A RAG query from the user."""
    query: str
    max_chunks: int = 5

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        if len(v) > 2000:
            raise ValueError("Query too long (max 2000 chars)")
        return v


class ChunkResult(BaseModel):
    """A single retrieved document chunk."""
    id: str
    content: str
    source: str
    similarity: float
    role_access: list[RoleType]
    department: str


class QueryResponse(BaseModel):
    """Full RAG response returned to the user."""
    answer: str
    chunks_used: list[ChunkResult]
    access_granted: bool
    latency_ms: float


# ---- Error Schemas ----

class ErrorResponse(BaseModel):
    """Standard error format across all endpoints."""
    detail: str
    error_code: Optional[str] = None


class ExtractedMetadataResponse(BaseModel):
    """Response after auto-extracting metadata from a document."""
    department:    str
    sensitivity:   str
    document_type: str
    role_access:   list[RoleType]
    topics:        list[str]
    summary:       str
    pii_detected:  bool
    confidence:    float
    needs_review:  bool
    review_reason: str


class MetadataReviewRequest(BaseModel):
    """
    Admin approves or modifies auto-extracted metadata.
    Sent when human reviews a flagged document.
    """
    document_id: str
    role_access: list[RoleType]
    department:  str
    approved:    bool = True
    notes:       str  = ""


class AutoIngestRequest(BaseModel):
    """
    Request to upload + auto-tag + ingest in one step.
    This is the production workflow — no manual metadata.
    """
    auto_approve_threshold: float = 0.85
    department_hint: str = ""   # optional hint from uploader
