"""Content expansion service for enriching document content."""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.config import settings
from app.services.generation.prompt_builder import PromptBuilder
from app.services.generation.document_builder import DocumentBuilder
from app.services.generation.rag import GenerationRAGService as KnowledgeRetrievalService
import logging

logger = logging.getLogger(__name__)


class ContentExpander:
    """Service for safely expanding document content to meet minimum requirements."""

    def __init__(self):
        """Initialize the content expander."""
        self.llm = ChatOpenAI(
            model=settings.gemini_model or "google/gemini-2.0-flash-exp",
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
            temperature=0,
        )
        self.prompt_builder = PromptBuilder()
        self.document_builder = DocumentBuilder()

    @staticmethod
    def _is_tiptap_format(document: Dict[str, Any]) -> bool:
        """Check if document is in TipTap format."""
        return document.get("type") == "doc" and "content" in document

    @staticmethod
    def _get_pages_from_tiptap(document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract pages from TipTap document format."""
        if not ContentExpander._is_tiptap_format(document):
            return []
        
        content = document.get("content", [])
        pages = []
        for node in content:
            if isinstance(node, dict) and node.get("type") == "page":
                pages.append(node)
        return pages

    @staticmethod
    def _count_pages_from_tiptap(document: Dict[str, Any]) -> int:
        """Count pages from TipTap format."""
        if ContentExpander._is_tiptap_format(document):
            return len(ContentExpander._get_pages_from_tiptap(document))
        else:
            # Legacy format
            return len(document.get("pages", []))

    async def expand_document(
        self,
        document: Dict[str, Any],
        requirements: Dict[str, Any],
        knowledge_chunks: List[Dict[str, Any]],
        validation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Expand document content to meet minimum requirements.
        
        Args:
            document: Current document structure
            requirements: Document requirements
            knowledge_chunks: Retrieved knowledge chunks for legal compliance
            validation_result: Validation results showing what needs expansion
        
        Returns:
            Expanded document structure
        """
        logger.info("Starting document content expansion...")
        
        try:
            # Build expansion prompt
            expansion_prompt = self._build_expansion_prompt(
                document, requirements, knowledge_chunks, validation_result
            )
            
            # Call LLM to expand content with streaming (with fallback)
            message = HumanMessage(content=expansion_prompt)
            response_text = ""
            chunk_count = 0
            try:
                async for chunk in self.llm.astream([message]):
                    chunk_content = None
                    if hasattr(chunk, "content"):
                        chunk_content = chunk.content
                    elif isinstance(chunk, str):
                        chunk_content = chunk
                    elif hasattr(chunk, "text"):
                        chunk_content = chunk.text
                    
                    if chunk_content:
                        response_text += chunk_content
                        chunk_count += 1
                
                # Fallback if streaming returned empty
                if not response_text:
                    logger.warning("Streaming returned empty response, falling back to ainvoke")
                    response = await self.llm.ainvoke([message])
                    response_text = response.content if hasattr(response, "content") else str(response)
                    logger.info(f"LLM expansion fallback response received, length: {len(response_text)} characters")
                else:
                    logger.info(f"LLM expansion streaming response received, length: {len(response_text)} characters ({chunk_count} chunks)")
            except Exception as stream_error:
                # Fallback to ainvoke if streaming fails
                logger.warning(f"Streaming failed: {stream_error}, falling back to ainvoke")
                response = await self.llm.ainvoke([message])
                response_text = response.content if hasattr(response, "content") else str(response)
                logger.info(f"LLM expansion fallback response received, length: {len(response_text)} characters")
            
            # Parse expanded document
            expanded_document = self.document_builder.parse_llm_output(response_text)
            
            # Count pages from TipTap format
            original_pages = self._count_pages_from_tiptap(document)
            expanded_pages = self._count_pages_from_tiptap(expanded_document)
            
            logger.info(
                f"Document expanded: {original_pages} -> {expanded_pages} pages"
            )
            
            return expanded_document
            
        except Exception as e:
            logger.error(f"Failed to expand document: {e}", exc_info=True)
            # Return original document if expansion fails
            return document
    
    def _build_expansion_prompt(
        self,
        document: Dict[str, Any],
        requirements: Dict[str, Any],
        knowledge_chunks: List[Dict[str, Any]],
        validation_result: Dict[str, Any],
    ) -> str:
        """Build prompt for content expansion."""
        # Format knowledge context
        knowledge_context = KnowledgeRetrievalService.format_knowledge_for_prompt(knowledge_chunks)
        
        # Format validation issues
        issues_text = "\n".join([f"- {issue}" for issue in validation_result.get("issues", [])])
        
        # Get language for appropriate example
        from app.services.generation.language import normalize_language_code

        language = normalize_language_code(requirements.get("language", "en"))
        if language == "am":
            structure_example = self.prompt_builder.DOCUMENT_STRUCTURE_EXAMPLE_AMHARIC
            language_instruction = """
CRITICAL LANGUAGE REQUIREMENT:
- You MUST expand the document in Amharic (አማርኛ)
- All new text, titles, content, clauses MUST be in Amharic
- Use Amharic characters and proper Amharic formatting
- Preserve the existing Amharic content
- The metadata language field MUST remain "am"
- IMPORTANT: Preserve user-provided information (names, addresses, dates, etc.) exactly as they appear
- Do NOT translate user-provided names, addresses, or specific information - keep them as-is
"""
        elif language == "om":
            structure_example = self.prompt_builder.DOCUMENT_STRUCTURE_EXAMPLE_ENGLISH
            language_instruction = """
CRITICAL LANGUAGE REQUIREMENT:
- You MUST expand the document in Afaan Oromo
- All new text, titles, content, clauses MUST be in Afaan Oromo
- Preserve the existing Afaan Oromo content
- The metadata language field MUST remain "om"
- IMPORTANT: Preserve user-provided information (names, addresses, dates, etc.) exactly as they appear
- Do NOT translate user-provided names, addresses, or specific information - keep them as-is
"""
        else:
            structure_example = self.prompt_builder.DOCUMENT_STRUCTURE_EXAMPLE_ENGLISH
            language_instruction = """
CRITICAL LANGUAGE REQUIREMENT:
- You MUST expand the document in English
- All new text, titles, content, clauses MUST be in English
- Preserve the existing English content
- The metadata language field MUST remain "en"
- IMPORTANT: Preserve user-provided information (names, addresses, dates, etc.) exactly as they appear
- Keep all user-provided information in its original language
"""
        
        # Get user-specified page count or use default
        target_pages = requirements.get("num_pages")
        if target_pages and isinstance(target_pages, int) and 1 <= target_pages <= 20:
            min_pages = target_pages
            max_pages = min(target_pages + 1, settings.MAX_PAGES)  # Allow 1 page buffer, but cap at max
        else:
            min_pages = settings.MIN_PAGES
            max_pages = settings.MAX_PAGES
        
        prompt = f"""You are a legal document expansion assistant. Your task is to expand and enrich an existing legal document to ensure it meets minimum requirements while maintaining legal compliance and accuracy.

{language_instruction}

CURRENT DOCUMENT:
{self._format_document_summary(document)}

VALIDATION ISSUES:
{issues_text}

REQUIREMENTS:
{self.prompt_builder._format_requirements(requirements)}

{knowledge_context}

TASK:
1. Expand the document to have at least {min_pages} pages but DO NOT exceed {max_pages} pages (currently it has fewer pages or insufficient content)
2. Ensure each page has at least {settings.MIN_WORDS_PER_PAGE} words of rich, detailed content
3. Add MORE sections, clauses, and detailed legal provisions - be THOROUGH and COMPREHENSIVE within the page limit
4. Use the legal knowledge provided above to ensure compliance
5. Maintain the exact TipTap document structure shown below
6. Preserve all existing content, block_ids, and structure
7. Add new sections/clauses with unique sequential block_ids (continue from existing IDs)
8. Ensure comprehensive legal coverage with detailed explanations
9. CRITICAL: Do NOT exceed {max_pages} pages - this is a hard limit for initial generation

DOCUMENT STRUCTURE (maintain this exact TipTap JSON format):
{structure_example}

EXPANSION STRATEGY - BE THOROUGH AND DETAILED:
- EXPAND existing paragraphs with more detailed explanations, examples, and legal precision
- Add missing standard legal clauses (definitions, obligations, termination, dispute resolution, governing law, etc.)
- Enhance definitions with comprehensive legal precision and examples
- Add procedural details, timelines, and step-by-step processes
- Include standard legal provisions relevant to the document type
- Add more pages if needed to reach minimum requirements, but do not exceed {max_pages} pages
- Break long sections into multiple detailed paragraphs
- Add sub-clauses and detailed provisions within existing sections
- Ensure all content is legally accurate, compliant, and THOROUGH

IMPORTANT:
- Do NOT remove or modify existing content
- Only ADD new content and EXPAND existing content with more detail
- Maintain all existing block_ids (b1, b2, etc.)
- Generate new unique sequential block_ids for new content (continue numbering)
- Use TipTap JSON structure (NOT HTML) - paragraphs, headings, etc. as JSON nodes
- Ensure legal accuracy using the knowledge base provided
- Make the document DENSE with detailed legal provisions

TIPTAP STRUCTURE RULES:
- Root: {{"type": "doc", "content": [...]}}
- Page: {{"type": "page", "content": [...]}}
- PageBreak: {{"type": "pageBreak"}} (between pages)
- Heading: {{"type": "heading", "attrs": {{"level": N}}, "content": [{{"type": "text", "text": "..."}}]}}
- Paragraph: {{"type": "paragraph", "attrs": {{"block_id": "bN"}}, "content": [{{"type": "text", "text": "..."}}]}}
- Text: {{"type": "text", "text": "your text here"}}
- Assign sequential block_ids: b1, b2, b3, etc. (continue from existing IDs)

CRITICAL JSON FORMATTING REQUIREMENTS:
- You MUST return ONLY valid JSON - no markdown, no code blocks, no explanatory text
- ALL double quotes inside string values MUST be escaped with backslash: \\"
- Example: text with quotes must be: {{"type": "text", "text": "The \\"Party\\" agrees..."}}
- NEVER use unescaped double quotes inside JSON string values
- No trailing commas: "key": "value",}} is WRONG, use "key": "value"}}
- All strings must use double quotes (not single quotes)
- Escape these characters in strings: \\" (quote), \\\\ (backslash), \\n (newline), \\t (tab)
- Ensure all brackets [ ] and braces {{ }} are properly closed and matched
- No comments anywhere in the JSON
- Every comma must be followed by a key, value, or closing bracket/brace
- The JSON must be valid and parseable by standard JSON parsers

Respond with ONLY the complete expanded TipTap JSON document structure - no markdown code blocks, no json tags, no additional text - just pure valid JSON starting with {{ and ending with }}.
"""
        return prompt
    
    def _format_document_summary(self, document: Dict[str, Any]) -> str:
        """Format document summary for prompt - works with both TipTap and legacy formats."""
        if self._is_tiptap_format(document):
            return self._format_tiptap_summary(document)
        else:
            return self._format_legacy_summary(document)
    
    def _format_tiptap_summary(self, document: Dict[str, Any]) -> str:
        """Format TipTap document summary for prompt."""
        pages = self._get_pages_from_tiptap(document)
        page_count = len(pages)
        
        summary = f"Document has {page_count} pages (TipTap format):\n"
        
        def extract_text_from_node(node: Dict[str, Any], max_length: int = 100) -> str:
            """Extract text from a TipTap node, truncated."""
            text_parts = []
            
            def collect_text(n: Dict[str, Any]):
                if not isinstance(n, dict):
                    return
                if n.get("type") == "text":
                    text_parts.append(n.get("text", ""))
                elif "content" in n:
                    for child in n.get("content", []):
                        collect_text(child)
            
            collect_text(node)
            text = " ".join(text_parts)
            if len(text) > max_length:
                return text[:max_length] + "..."
            return text
        
        for i, page in enumerate(pages, 1):
            page_content = page.get("content", [])
            summary += f"\nPage {i}:\n"
            
            block_count = 0
            for node in page_content:
                if not isinstance(node, dict):
                    continue
                
                node_type = node.get("type", "")
                if node_type == "heading":
                    level = node.get("attrs", {}).get("level", 1)
                    text = extract_text_from_node(node, 80)
                    summary += f"  - Heading (level {level}): {text}\n"
                elif node_type == "paragraph":
                    block_id = node.get("attrs", {}).get("block_id", "")
                    text = extract_text_from_node(node, 80)
                    summary += f"  - Paragraph {block_id}: {text}\n"
                    block_count += 1
            
            summary += f"  (Total blocks: {block_count})\n"
        
        return summary
    
    def _format_legacy_summary(self, document: Dict[str, Any]) -> str:
        """Format legacy document summary for prompt."""
        pages = document.get("pages", [])
        page_count = len(pages)
        
        summary = f"Document has {page_count} pages:\n"
        
        for i, page in enumerate(pages, 1):
            sections = page.get("sections", [])
            summary += f"\nPage {i}:\n"
            for section in sections:
                section_title = section.get("title", "")
                section_type = section.get("type", "")
                clauses = section.get("clauses", [])
                summary += f"  - {section_type}: {section_title} ({len(clauses)} clauses)\n"
        
        return summary

