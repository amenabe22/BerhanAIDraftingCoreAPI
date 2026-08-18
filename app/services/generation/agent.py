"""Document generation agent (ported from main API; SSE + in-memory threads)."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage

from app.config import settings
from app.llm import build_chat_llm, resolve_model
from app.services.generation.document_builder import DocumentBuilder
from app.services.generation.document_validator import DocumentValidator
from app.services.generation.error_utils import sanitize_error_message
from app.services.generation.json_syntax_checker import JSONSyntaxChecker
from app.services.generation.json_utils import parse_json
from app.services.generation.language import normalize_language_code
from app.services.generation.prompt_builder import PromptBuilder
from app.services.generation.rag import GenerationRAGService
from app.services.generation.schemas import (
    ANALYSIS_RESPONSE_SCHEMA,
    CLARIFICATION_RESPONSE_SCHEMA,
)
from app.services.generation.sse import SSEEmitter
from app.services.generation.streaming_json_validator import (
    StreamingJSONValidator,
    ValidationErrorType,
)
from app.services.generation.streaming_parser import TipTapStreamingParser
from app.services.generation.thread_store import (
    ConversationStatus,
    ThreadStore,
    thread_store,
)

logger = logging.getLogger(__name__)


def _human_content(message: str, file_url: str | None) -> str | list[dict[str, Any]]:
    """Build LangChain human content, optionally multimodal via file_url."""
    url = (file_url or "").strip()
    if not url:
        return message
    return [
        {"type": "text", "text": message},
        {"type": "image_url", "image_url": {"url": url}},
    ]


class GenerationAgent:
    """Analyze requirements and generate TipTap documents over SSE."""

    def __init__(self, store: ThreadStore | None = None) -> None:
        self.store = store or thread_store
        self.prompt_builder = PromptBuilder()
        self.document_builder = DocumentBuilder()
        self.document_validator = DocumentValidator()
        self.knowledge_retrieval = GenerationRAGService()
        self.llm_timeout = int(
            os.getenv("DRAFTING_LLM_TIMEOUT", str(settings.DRAFTING_LLM_TIMEOUT))
        )
        self.knowledge_timeout = int(
            os.getenv("DRAFTING_KNOWLEDGE_TIMEOUT", str(settings.DRAFTING_KNOWLEDGE_TIMEOUT))
        )
        self.progress_update_interval = 50
        self.max_knowledge_chunks = settings.GENERATION_KNOWLEDGE_TOP_K

    def _llm(self, model: str | None, enable_reasoning: bool = False):
        resolved = resolve_model(model)
        return build_chat_llm(
            model=resolved,
            enable_reasoning=enable_reasoning,
            streaming=True,
        ), resolved

    async def analyze_requirements(
        self,
        thread_id: str,
        user_prompt: str,
        emitter: SSEEmitter,
        *,
        context: dict[str, Any] | None = None,
        model: str | None = None,
        enable_reasoning: bool = False,
        file_url: str | None = None,
    ) -> dict[str, Any]:
        try:
            await emitter.analyzing(thread_id)
            await emitter.status("Analyzing document requirements…")

            state = self.store.get(thread_id)
            if not state:
                raise ValueError(f"Thread {thread_id} not found")

            history = self.store.history(state)
            prompt = self.prompt_builder.build_analysis_prompt(
                user_prompt, history, context or {}
            )
            llm, _ = self._llm(model, enable_reasoning)
            message = HumanMessage(content=_human_content(prompt, file_url))

            response_text = ""
            try:
                async for chunk in llm.astream([message]):
                    chunk_content = getattr(chunk, "content", None) or (
                        chunk if isinstance(chunk, str) else None
                    )
                    if chunk_content:
                        response_text += chunk_content
                        if isinstance(chunk_content, str):
                            await emitter.token(chunk_content)
                if not response_text:
                    response = await llm.ainvoke([message])
                    response_text = (
                        response.content if hasattr(response, "content") else str(response)
                    )
            except Exception:
                response = await llm.ainvoke([message])
                response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )

            preferred_language = (context or {}).get("language")

            try:
                analysis = parse_json(
                    response_text,
                    schema=ANALYSIS_RESPONSE_SCHEMA,
                    extract_from_markdown=True,
                )
            except ValueError:
                logger.warning("Failed to parse analysis response, using fallback")
                analysis = {
                    "ready_to_generate": False,
                    "response_message": response_text,
                    "questions": [
                        "Could you provide more details about the document you need?"
                    ],
                    "document_type": "contract",
                    "language": preferred_language or "en",
                }

            self.store.add_user_message(thread_id, user_prompt)
            self.store.add_system_response(
                thread_id, analysis.get("response_message", "")
            )

            extracted = dict(analysis.get("extracted_info") or {})
            # Seed requirements from analysis + request context.
            # User/agent language preference MUST win over LLM guesses
            # (same pin as BerhanAdvisorCoreAPI drafting generation endpoint).
            reqs: dict[str, Any] = {
                "document_type": analysis.get("document_type", "contract"),
                **extracted,
            }
            if preferred_language:
                pinned = normalize_language_code(preferred_language)
                reqs["language"] = pinned
                analysis["language"] = pinned
                extracted["language"] = pinned
                analysis["extracted_info"] = extracted
            else:
                reqs["language"] = normalize_language_code(
                    analysis.get("language") or extracted.get("language") or "en"
                )
            if context and context.get("num_pages"):
                reqs["num_pages"] = context["num_pages"]
            if context and context.get("document_type"):
                reqs["document_type"] = context["document_type"]
            self.store.set_requirements(thread_id, reqs)

            return analysis
        except Exception as e:
            logger.error("Error analyzing requirements: %s", e, exc_info=True)
            raise ValueError(sanitize_error_message(e)) from e

    async def process_clarification(
        self,
        thread_id: str,
        user_message: str,
        emitter: SSEEmitter,
        *,
        model: str | None = None,
        enable_reasoning: bool = False,
        file_url: str | None = None,
    ) -> dict[str, Any]:
        try:
            await emitter.status("Processing clarification…")
            state = self.store.get(thread_id)
            if not state:
                raise ValueError(f"Thread {thread_id} not found")

            language = state.extracted_requirements.get("language", "en")
            history = self.store.history(state)
            prompt = self.prompt_builder.build_clarification_prompt(
                user_message, history, language
            )
            llm, _ = self._llm(model, enable_reasoning)
            message = HumanMessage(content=_human_content(prompt, file_url))
            response = await llm.ainvoke([message])
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            try:
                result = parse_json(
                    response_text,
                    schema=CLARIFICATION_RESPONSE_SCHEMA,
                    extract_from_markdown=True,
                )
            except ValueError:
                logger.warning("Failed to parse clarification response, using fallback")
                result = {
                    "ready_to_generate": False,
                    "response_message": response_text,
                    "questions": [],
                }

            self.store.add_user_message(thread_id, user_message)
            self.store.add_system_response(
                thread_id, result.get("response_message", "")
            )
            if result.get("updated_requirements"):
                updates = dict(result["updated_requirements"])
                # Keep the pinned preference; do not let the LLM flip language mid-thread.
                preferred = (state.extracted_requirements or {}).get("language")
                if preferred:
                    updates["language"] = normalize_language_code(preferred)
                self.store.update_requirements(thread_id, updates)
            return result
        except Exception as e:
            logger.error("Error processing clarification: %s", e, exc_info=True)
            raise ValueError(sanitize_error_message(e)) from e

    async def generate_from_requirements(
        self,
        thread_id: str,
        emitter: SSEEmitter,
        *,
        requirements: dict[str, Any],
        synthetic_prompt: str,
        model: str | None = None,
        enable_reasoning: bool = False,
    ) -> dict[str, Any]:
        """One-shot generation from a pre-built requirements payload (no clarification)."""
        state = self.store.get(thread_id)
        if not state:
            raise ValueError(f"Thread {thread_id} not found")

        pinned = dict(requirements or {})
        if pinned.get("language"):
            pinned["language"] = normalize_language_code(pinned["language"])
        self.store.set_requirements(thread_id, pinned)
        self.store.add_user_message(thread_id, synthetic_prompt)
        self.store.add_system_response(
            thread_id,
            "Requirements received. Generating the document without further questions.",
        )
        await emitter.ready_to_generate(thread_id, "Generating from structured requirements…")
        return await self.generate_document(
            thread_id,
            emitter,
            model=model,
            enable_reasoning=enable_reasoning,
        )

    async def generate_document(
        self,
        thread_id: str,
        emitter: SSEEmitter,
        *,
        model: str | None = None,
        enable_reasoning: bool = False,
    ) -> dict[str, Any]:
        start_time = time.time()
        try:
            await emitter.status("Starting document generation…")
            await emitter.generation_progress(
                thread_id, 0.1, "preparing_generation", "Preparing to generate document..."
            )

            state = self.store.get(thread_id)
            if not state:
                raise ValueError(f"Thread {thread_id} not found")

            self.store.update_status(thread_id, ConversationStatus.GENERATING)
            requirements = dict(state.extracted_requirements or {})
            if requirements.get("language"):
                requirements["language"] = normalize_language_code(
                    requirements["language"]
                )
                self.store.update_requirements(
                    thread_id, {"language": requirements["language"]}
                )
            history = self.store.history(state)

            await emitter.generation_progress(
                thread_id,
                0.15,
                "retrieving_knowledge",
                "Retrieving relevant legal knowledge...",
            )

            knowledge_chunks: list[dict[str, Any]] = []
            citations: list[dict[str, Any]] = []
            laws_referenced: list[str] = []
            knowledge_context = ""

            try:
                knowledge_chunks = await asyncio.wait_for(
                    self.knowledge_retrieval.retrieve_relevant_knowledge(requirements),
                    timeout=self.knowledge_timeout,
                )
                knowledge_chunks = knowledge_chunks[: self.max_knowledge_chunks]
            except asyncio.TimeoutError:
                logger.warning(
                    "[%s] Knowledge retrieval timed out after %ss",
                    thread_id,
                    self.knowledge_timeout,
                )
            except Exception as e:
                logger.warning("[%s] Knowledge retrieval failed: %s", thread_id, e)

            if knowledge_chunks:
                citations = self.knowledge_retrieval.extract_citations(knowledge_chunks)
                laws_referenced = self.knowledge_retrieval.extract_law_references(
                    knowledge_chunks
                )
                await emitter.citations(
                    thread_id,
                    citations,
                    laws_referenced,
                    message=(
                        f"Found {len(citations)} legal sources and "
                        f"{len(laws_referenced)} law references"
                    ),
                )
                knowledge_context = self.knowledge_retrieval.format_knowledge_for_prompt(
                    knowledge_chunks
                )

            await emitter.generation_progress(
                thread_id, 0.3, "building_prompt", "Building generation prompt..."
            )
            prompt = self.prompt_builder.build_generation_prompt(
                requirements, history, knowledge_context
            )

            await emitter.generation_progress(
                thread_id,
                0.5,
                "generating_content",
                "Generating document content with AI...",
            )

            llm, _ = self._llm(model, enable_reasoning)
            # JSON object mode when supported
            try:
                llm = llm.bind(response_format={"type": "json_object"})
            except Exception:
                pass

            message = HumanMessage(content=prompt)
            response_text = ""
            chunk_count = 0
            last_progress_update = 0
            nodes_streamed = 0
            parser = TipTapStreamingParser()
            validator = StreamingJSONValidator()
            llm_start = time.time()

            async def stream_llm() -> None:
                nonlocal response_text, chunk_count, last_progress_update, nodes_streamed
                async for chunk in llm.astream([message]):
                    chunk_content = None
                    if hasattr(chunk, "content"):
                        chunk_content = chunk.content
                    elif isinstance(chunk, str):
                        chunk_content = chunk
                    if not chunk_content:
                        continue
                    if isinstance(chunk_content, list):
                        # Multimodal / content-block responses
                        texts = []
                        for part in chunk_content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                texts.append(part.get("text", ""))
                            elif isinstance(part, str):
                                texts.append(part)
                        chunk_content = "".join(texts)
                    if not chunk_content:
                        continue

                    response_text += chunk_content
                    chunk_count += 1
                    await emitter.token(chunk_content)

                    validation_errors = validator.feed(chunk_content)
                    for error in validation_errors:
                        if error.error_type == ValidationErrorType.MISSING_COMMA:
                            logger.warning(
                                "[%s] JSON validation warning: %s",
                                thread_id,
                                error.message,
                            )

                    try:
                        complete_nodes = parser.feed(chunk_content)
                        for node in complete_nodes:
                            nodes_streamed += 1
                            await emitter.node_stream(thread_id, node, nodes_streamed)
                    except Exception as parse_error:
                        logger.debug(
                            "[%s] Parser error during streaming: %s",
                            thread_id,
                            parse_error,
                        )

                    if chunk_count - last_progress_update >= self.progress_update_interval:
                        progress = 0.5 + min(0.3, len(response_text) / 50000 * 0.3)
                        await emitter.generation_progress(
                            thread_id,
                            progress,
                            "generating_content",
                            (
                                f"Streaming document... ({nodes_streamed} nodes, "
                                f"{len(response_text)} characters)"
                            ),
                        )
                        last_progress_update = chunk_count

            try:
                await asyncio.wait_for(stream_llm(), timeout=self.llm_timeout)
            except asyncio.TimeoutError:
                if len(response_text) < 100:
                    raise ValueError(
                        f"Document generation timed out after {self.llm_timeout} seconds "
                        "with minimal output."
                    ) from None
                logger.warning(
                    "[%s] Continuing with partial response (%s chars)",
                    thread_id,
                    len(response_text),
                )
            except Exception as stream_error:
                logger.warning(
                    "[%s] Streaming failed: %s, falling back to ainvoke",
                    thread_id,
                    stream_error,
                    exc_info=True,
                )
                response = await asyncio.wait_for(
                    llm.ainvoke([message]), timeout=self.llm_timeout
                )
                response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )

            if not response_text:
                response = await asyncio.wait_for(
                    llm.ainvoke([message]), timeout=self.llm_timeout
                )
                response_text = (
                    response.content if hasattr(response, "content") else str(response)
                )

            await emitter.generation_progress(
                thread_id,
                0.7,
                "parsing_structure",
                "Parsing and validating document structure...",
            )

            document = None
            parse_attempts = 0
            max_retries = 2
            while parse_attempts <= max_retries:
                try:
                    document = self.document_builder.parse_llm_output(response_text)
                    break
                except ValueError as parse_error:
                    parse_attempts += 1
                    error_msg = str(parse_error)
                    if parse_attempts > max_retries:
                        syntax_checker = JSONSyntaxChecker()
                        _ok, syntax_errors = syntax_checker.check_syntax(response_text)
                        enhanced = error_msg
                        if syntax_errors:
                            missing = sum(
                                1 for e in syntax_errors if e.error_type == "missing_comma"
                            )
                            if missing:
                                enhanced = (
                                    f"{error_msg}\n\nDetected {missing} missing comma(s)."
                                )
                        raise ValueError(enhanced) from parse_error

                    error_pos = None
                    pos_match = re.search(r"char (\d+)", error_msg)
                    if pos_match:
                        error_pos = int(pos_match.group(1))
                    retry_prompt = self._build_retry_prompt(
                        original_prompt=prompt,
                        error_message=error_msg,
                        error_position=error_pos,
                        response_text=response_text,
                    )
                    retry_timeout = max(30, self.llm_timeout // 2)
                    retry_message = HumanMessage(content=retry_prompt)
                    response = await asyncio.wait_for(
                        llm.ainvoke([retry_message]), timeout=retry_timeout
                    )
                    response_text = (
                        response.content if hasattr(response, "content") else str(response)
                    )
                    validator.reset()
                    validator.feed(response_text)

            assert document is not None

            # Ensure TipTap metadata.language matches the pinned preference
            preferred_lang = normalize_language_code(requirements.get("language", "en"))
            meta = document.get("metadata")
            if isinstance(meta, dict):
                meta = dict(meta)
                meta["language"] = preferred_lang
                document["metadata"] = meta
            else:
                document["metadata"] = {
                    "document_type": requirements.get("document_type", "Contract"),
                    "title": requirements.get("title")
                    or requirements.get("document_type", "Document"),
                    "language": preferred_lang,
                }

            await emitter.generation_progress(
                thread_id, 0.9, "validating_document", "Validating streamed document..."
            )
            validation_result = self.document_validator.validate_document(
                document, requirements
            )
            if not validation_result.get("valid"):
                logger.warning(
                    "[%s] Document validation issues (accepting): %s",
                    thread_id,
                    validation_result.get("issues", []),
                )

            document["_citations"] = citations
            document["_laws_referenced"] = laws_referenced
            self.store.update_document(thread_id, document)
            self.store.update_status(thread_id, ConversationStatus.COMPLETED)

            await emitter.generation_progress(
                thread_id, 1.0, "completed", "Document structure generated successfully!"
            )
            await emitter.document_generated(
                thread_id,
                document,
                metadata={
                    "page_count": validation_result.get("page_count"),
                    "total_words": validation_result.get("total_words"),
                    "duration_s": round(time.time() - start_time, 2),
                    "citations": citations,
                    "laws_referenced": laws_referenced,
                },
            )
            return document
        except Exception as e:
            logger.error("Error generating document: %s", e, exc_info=True)
            self.store.update_status(thread_id, ConversationStatus.ERROR)
            raise ValueError(sanitize_error_message(e)) from e

    def _build_retry_prompt(
        self,
        original_prompt: str,
        error_message: str,
        error_position: int | None = None,
        response_text: str = "",
    ) -> str:
        error_context = ""
        if error_position is not None and error_position < len(response_text):
            start = max(0, error_position - 100)
            end = min(len(response_text), error_position + 100)
            ctx = response_text[start:end]
            rel = error_position - start
            error_context = ctx[:rel] + " ⬅️ ERROR HERE ⬅️ " + ctx[rel:]

        return f"""{original_prompt}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RETRY: Your previous JSON could not be parsed.

Error: {error_message}
{f"Context: {error_context}" if error_context else ""}

Fix the JSON. After every }} or ]] that is NOT the last element, add a comma.
Return ONLY valid TipTap JSON. No markdown fences.
"""
