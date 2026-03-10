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

    # Compliance analysis
    COMPLIANCE_ANALYSIS_MODEL: str | None = None  # default: GEMINI_MODEL
    COMPLIANCE_RERANKER_MODEL: str | None = None  # default: same as analysis model
    COMPLIANCE_IMPLICATION_INITIAL_LIMIT: int = 15
    COMPLIANCE_MAX_CLAUSES_FOR_CITATIONS: int = 20
    COMPLIANCE_ANALYSIS_TEMPERATURE: float = 0.1
    COMPLIANCE_ANALYSIS_MAX_TOKENS: int = (
        16384  # allow long analysis JSON (clauses/issues); avoid truncation
    )
    COMPLIANCE_SCORE_ROUNDING: int = 2
    # Normalization ceiling for the deterministic scoring engine.
    # raw_penalty is expressed as a fraction of this value to produce risk_score 0–100.
    # Raise to make the scale more forgiving; lower to make it stricter.
    COMPLIANCE_SCORE_MAX_PENALTY: int = 150
    KNOWLEDGE_RERANK_INITIAL_LIMIT: int = 25
    KNOWLEDGE_RERANK_TOP: int = 10

    @field_validator("QDRANT_API_KEY", "COHERE_API_KEY", mode="before")
    @classmethod
    def strip_api_key_comment(cls, v: str | None) -> str | None:
        return _strip_comment(v)


settings = Settings()
