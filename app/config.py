from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _strip_comment(v: str | None) -> str | None:
    if v is None or not isinstance(v, str):
        return v
    return v.split("#")[0].strip() or None if v.strip() else None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Qdrant (payload keys must match how the collection was indexed)
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    QDRANT_VERIFY_SSL: bool = True  # set to false to skip TLS cert verification
    QDRANT_LEGAL_KNOWLEDGE_COLLECTION: str = "BerhanAIDocumentKnowledgeCited"
    QDRANT_DEFAULT_COLLECTION: str = "doc_blocks"  # user-uploaded doc blocks
    KNOWLEDGE_EMBEDDING_DIMENSION: int = 1024
    QDRANT_CONTENT_PAYLOAD_KEY: str = "content"  # key in point payload for article text
    QDRANT_METADATA_PAYLOAD_KEY: str = (
        "metadata"  # key for nested metadata; if missing, metadata will be {}
    )

    # OpenRouter (LLM)
    GEMINI_MODEL: str = "google/gemini-2.5-flash"
    OPENROUTER_API_KEY: str = ""

    # Embeddings – Cohere (must match model used to index Qdrant; embed-multilingual-v3.0 is 1024-dim)
    COHERE_API_KEY: str | None = None
    COHERE_API_URL: str = "https://api.cohere.com"
    COHERE_EMBEDDING_MODEL: str = "embed-multilingual-v3.0"

    # Retrieval – number of chunks returned per query (legal search/advice vs doc search)
    RETRIEVAL_LEGAL_TOP_K: int = 12
    RETRIEVAL_DOC_TOP_K: int = 10
    # Chat-path legal retrieval: semantic pipeline (expand → RRF over-fetch → Cohere Rerank)
    RETRIEVAL_LEGAL_FETCH_K: int = 15      # candidates per sub-query fetched from Qdrant
    RETRIEVAL_LEGAL_RERANK_TOP_K: int = 12  # final results after Cohere cross-encoder rerank

    # Compliance analysis
    COMPLIANCE_ANALYSIS_MODEL: str | None = None  # default: GEMINI_MODEL
    COMPLIANCE_RERANKER_MODEL: str | None = None  # default: same as analysis model
    COMPLIANCE_IMPLICATION_INITIAL_LIMIT: int = 15
    COMPLIANCE_MAX_CLAUSES_FOR_CITATIONS: int = 20
    COMPLIANCE_ANALYSIS_TEMPERATURE: float = 0.0
    COMPLIANCE_ANALYSIS_SEED: int = 7
    COMPLIANCE_ANALYSIS_MAX_TOKENS: int = (
        32768  # editor_fix per clause increases output size; avoid truncation
    )
    # Reasoning effort for main rubric evaluation only (low|medium|high; empty = off)
    COMPLIANCE_REASONING_EFFORT: str = ""
    COMPLIANCE_SCORE_ROUNDING: int = 2
    # Normalization ceiling for the deterministic scoring engine.
    # raw_penalty is expressed as a fraction of this value to produce risk_score 0–100.
    # Raise to make the scale more forgiving; lower to make it stricter.
    COMPLIANCE_SCORE_MAX_PENALTY: int = 150
    KNOWLEDGE_RERANK_INITIAL_LIMIT: int = 25
    KNOWLEDGE_RERANK_TOP: int = 10

    # Semantic edit agent
    EDIT_MODEL: str | None = None  # default: GEMINI_MODEL
    COHERE_RERANK_MODEL: str = "rerank-multilingual-v3.0"
    EDIT_RERANK_TOP_K: int = 12
    EDIT_SELECTOR_TOP_K: int = 10
    EDIT_MIN_CONFIDENCE: float = 0.55
    EDIT_MAX_REVISIONS: int = 2

    # Redis (compliance result cache + diff anchoring)
    REDIS_URL: str = "redis://localhost:6379/0"
    COMPLIANCE_CACHE_TTL: int = 604_800  # 7 days

    # Document generation (ported from main API drafting)
    MIN_PAGES: int = 4
    MAX_PAGES: int = 6
    MIN_WORDS_PER_PAGE: int = 500
    DRAFTING_LLM_TIMEOUT: int = 120
    DRAFTING_KNOWLEDGE_TIMEOUT: int = 10
    ENABLE_GENERATION_RAG: bool = True
    GENERATION_KNOWLEDGE_TOP_K: int = 3
    ENABLE_CONTENT_EXPANSION: bool = False

    # Contabo Object Storage (S3-compatible) for PDF/DOCX export
    S3_ENDPOINT_URL: str = ""
    S3_ACCESS_KEY_ID: str = ""
    S3_SECRET_ACCESS_KEY: str = ""
    S3_BUCKET_NAME: str = ""
    S3_REGION: str = "default"
    S3_PUBLIC_BASE_URL: str | None = None  # if set, public URLs; else presigned GET
    S3_PRESIGN_EXPIRY_SECONDS: int = 604_800  # 7 days

    @field_validator(
        "QDRANT_API_KEY",
        "COHERE_API_KEY",
        "S3_PUBLIC_BASE_URL",
        mode="before",
    )
    @classmethod
    def strip_optional_secret_comment(cls, v: str | None) -> str | None:
        return _strip_comment(v)

    @field_validator(
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_ENDPOINT_URL",
        "S3_BUCKET_NAME",
        mode="before",
    )
    @classmethod
    def strip_s3_string_comment(cls, v: str | None) -> str:
        if v is None:
            return ""
        if not isinstance(v, str):
            return str(v)
        return v.split("#")[0].strip()



settings = Settings()
