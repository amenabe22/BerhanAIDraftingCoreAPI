"""Document validation service for ensuring minimum requirements."""

from typing import Dict, Any, List, Optional
from app.config import settings
import logging

logger = logging.getLogger(__name__)


class DocumentValidator:
    """Service for validating document structure and content requirements."""

    @staticmethod
    def _is_tiptap_format(document: Dict[str, Any]) -> bool:
        """Check if document is in TipTap format."""
        return document.get("type") == "doc" and "content" in document

    @staticmethod
    def _get_pages_from_tiptap(document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract pages from TipTap document format."""
        if not DocumentValidator._is_tiptap_format(document):
            return []
        
        content = document.get("content", [])
        pages = []
        for node in content:
            if isinstance(node, dict) and node.get("type") == "page":
                pages.append(node)
        return pages

    @staticmethod
    def _extract_text_from_tiptap_node(node: Dict[str, Any]) -> str:
        """Extract text content from a TipTap node recursively."""
        text_parts = []
        
        def collect_text(n: Dict[str, Any]):
            """Recursively collect text from node."""
            if not isinstance(n, dict):
                return
            if n.get("type") == "text":
                text_parts.append(n.get("text", ""))
            elif "content" in n:
                for child in n.get("content", []):
                    collect_text(child)
        
        collect_text(node)
        return " ".join(text_parts)

    @staticmethod
    def validate_document(document: Dict[str, Any], requirements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate document against minimum requirements.
        
        Args:
            document: Document structure dictionary (TipTap format or legacy format)
            requirements: Optional requirements dict that may contain num_pages
        
        Returns:
            Dictionary with validation results:
            - valid: bool
            - meets_min_pages: bool
            - meets_min_words: bool
            - page_count: int
            - total_words: int
            - issues: List[str]
            - needs_expansion: bool
        """
        issues = []
        
        # Get user-specified page count or use default
        target_pages = None
        if requirements:
            num_pages = requirements.get("num_pages")
            if isinstance(num_pages, int) and 1 <= num_pages <= 20:
                target_pages = num_pages
        
        min_pages = target_pages if target_pages else settings.MIN_PAGES
        max_pages = settings.MAX_PAGES
        
        # Handle TipTap format vs legacy format
        if DocumentValidator._is_tiptap_format(document):
            # TipTap format: extract pages from content array
            pages = DocumentValidator._get_pages_from_tiptap(document)
            page_count = len(pages)
        else:
            # Legacy format: use pages array directly
            pages = document.get("pages", [])
            page_count = len(pages)
        
        # Check minimum pages
        meets_min_pages = page_count >= min_pages
        if not meets_min_pages:
            issues.append(
                f"Document has {page_count} pages, minimum required is {min_pages}"
            )
        
        # Check maximum pages (for initial generation)
        meets_max_pages = page_count <= max_pages
        if not meets_max_pages:
            issues.append(
                f"Document has {page_count} pages, maximum allowed for initial generation is {max_pages}"
            )
        
        # Calculate total words
        total_words = DocumentValidator._count_words(document)
        min_total_words = min_pages * settings.MIN_WORDS_PER_PAGE
        meets_min_words = total_words >= min_total_words
        
        if not meets_min_words:
            issues.append(
                f"Document has {total_words} words, minimum required is {min_total_words} "
                f"({min_pages} pages × {settings.MIN_WORDS_PER_PAGE} words/page)"
            )
        
        # Check content richness
        avg_words_per_page = total_words / page_count if page_count > 0 else 0
        if avg_words_per_page < settings.MIN_WORDS_PER_PAGE:
            issues.append(
                f"Average words per page ({avg_words_per_page:.0f}) is below minimum "
                f"({settings.MIN_WORDS_PER_PAGE} words/page)"
            )
        
        # Check section depth and clause count
        section_count = DocumentValidator._count_sections(document)
        clause_count = DocumentValidator._count_clauses(document)
        
        if section_count < 5:
            issues.append(
                f"Document has {section_count} sections, should have at least 5 for comprehensive coverage"
            )
        
        if clause_count < 10:
            issues.append(
                f"Document has {clause_count} clauses, should have at least 10 for detailed legal coverage"
            )
        
        valid = meets_min_pages and meets_min_words and meets_max_pages and len(issues) == 0
        
        # Trigger expansion more aggressively:
        # 1. If document doesn't meet minimums (not valid)
        # 2. If there are quality issues (sections/clauses too low)
        # 3. If document is at minimum but could be more detailed (low word density)
        # 4. Don't expand if already at or above max_pages
        should_expand = (
            not valid or  # Doesn't meet minimums
            section_count < 5 or  # Too few sections
            clause_count < 10 or  # Too few clauses
            (page_count == min_pages and total_words < min_pages * settings.MIN_WORDS_PER_PAGE * 1.5)  # At minimum but not dense enough
        ) and page_count < max_pages  # Don't expand if already at max pages
        needs_expansion = should_expand and settings.ENABLE_CONTENT_EXPANSION
        
        result = {
            "valid": valid,
            "meets_min_pages": meets_min_pages,
            "meets_max_pages": meets_max_pages,
            "meets_min_words": meets_min_words,
            "page_count": page_count,
            "total_words": total_words,
            "avg_words_per_page": avg_words_per_page,
            "section_count": section_count,
            "clause_count": clause_count,
            "issues": issues,
            "needs_expansion": needs_expansion,
        }
        
        logger.info(
            f"Document validation: valid={valid}, pages={page_count}, "
            f"words={total_words}, sections={section_count}, clauses={clause_count}"
        )
        
        return result
    
    @staticmethod
    def _count_words(document: Dict[str, Any]) -> int:
        """Count total words in document."""
        word_count = 0
        
        # Count words in metadata title (if present)
        metadata = document.get("metadata", {})
        if metadata:
            title = metadata.get("title", "")
            if title:
                word_count += len(title.split())
        
        # Handle TipTap format vs legacy format
        if DocumentValidator._is_tiptap_format(document):
            # TipTap format: extract text from all nodes
            # Extract all text from the document and count words
            def extract_all_text(node: Dict[str, Any]) -> str:
                """Recursively extract all text from a TipTap node."""
                if not isinstance(node, dict):
                    return ""
                
                text_parts = []
                node_type = node.get("type", "")
                
                # Skip pageBreak nodes
                if node_type == "pageBreak":
                    return ""
                
                # Extract text from this node
                if node_type == "text":
                    text_parts.append(node.get("text", ""))
                elif "content" in node:
                    # Recursively extract from children
                    for child in node.get("content", []):
                        child_text = extract_all_text(child)
                        if child_text:
                            text_parts.append(child_text)
                
                return " ".join(text_parts)
            
            # Count words in all pages
            pages = DocumentValidator._get_pages_from_tiptap(document)
            for page in pages:
                page_content = page.get("content", [])
                for node in page_content:
                    if isinstance(node, dict):
                        text = extract_all_text(node)
                        if text:
                            word_count += len(text.split())
            
            # If no pages found, count from root content
            if not pages:
                content = document.get("content", [])
                for node in content:
                    if isinstance(node, dict):
                        text = extract_all_text(node)
                        if text:
                            word_count += len(text.split())
        else:
            # Legacy format: count words in pages/sections/clauses
            pages = document.get("pages", [])
            for page in pages:
                sections = page.get("sections", [])
                for section in sections:
                    # Count words in section title
                    section_title = section.get("title", "")
                    if section_title:
                        word_count += len(section_title.split())
                    
                    # Count words in section content (HTML)
                    content = section.get("content", "")
                    if content:
                        # Simple word count from HTML (remove tags)
                        import re
                        text = re.sub(r'<[^>]+>', ' ', content)
                        word_count += len(text.split())
                    
                    # Count words in clauses
                    clauses = section.get("clauses", [])
                    for clause in clauses:
                        clause_title = clause.get("title", "")
                        if clause_title:
                            word_count += len(clause_title.split())
                        
                        clause_content = clause.get("content", "")
                        if clause_content:
                            import re
                            text = re.sub(r'<[^>]+>', ' ', clause_content)
                            word_count += len(text.split())
        
        return word_count
    
    @staticmethod
    def _count_sections(document: Dict[str, Any]) -> int:
        """Count total sections in document."""
        count = 0
        
        # Handle TipTap format vs legacy format
        if DocumentValidator._is_tiptap_format(document):
            # TipTap format: count headings and paragraphs as sections
            def count_sections_in_node(node: Dict[str, Any]) -> int:
                """Count sections (headings and paragraphs) in a TipTap node."""
                if not isinstance(node, dict):
                    return 0
                
                section_count = 0
                node_type = node.get("type", "")
                
                # Count headings and paragraphs as sections
                if node_type in ["heading", "paragraph"]:
                    # Check if it has meaningful content
                    text = DocumentValidator._extract_text_from_tiptap_node(node)
                    if text.strip():
                        section_count += 1
                
                # Recursively count in children
                if "content" in node:
                    for child in node.get("content", []):
                        section_count += count_sections_in_node(child)
                
                return section_count
            
            # Count sections in all pages
            pages = DocumentValidator._get_pages_from_tiptap(document)
            for page in pages:
                page_content = page.get("content", [])
                for node in page_content:
                    if isinstance(node, dict):
                        count += count_sections_in_node(node)
            
            # If no pages found, count from root content
            if not pages:
                content = document.get("content", [])
                for node in content:
                    if isinstance(node, dict):
                        count += count_sections_in_node(node)
        else:
            # Legacy format: count sections from pages
            pages = document.get("pages", [])
            for page in pages:
                sections = page.get("sections", [])
                count += len(sections)
        
        return count
    
    @staticmethod
    def _count_clauses(document: Dict[str, Any]) -> int:
        """Count total clauses in document."""
        count = 0
        
        # Handle TipTap format vs legacy format
        if DocumentValidator._is_tiptap_format(document):
            # In TipTap format, clauses are typically represented as list items or nested paragraphs
            # For now, we'll count list items as clauses
            def count_clauses_in_node(node: Dict[str, Any]) -> int:
                """Count clauses (list items) in a TipTap node."""
                if not isinstance(node, dict):
                    return 0
                
                clause_count = 0
                node_type = node.get("type", "")
                
                # Count list items as clauses
                if node_type == "listItem":
                    text = DocumentValidator._extract_text_from_tiptap_node(node)
                    if text.strip():
                        clause_count += 1
                
                # Recursively count in children
                if "content" in node:
                    for child in node.get("content", []):
                        clause_count += count_clauses_in_node(child)
                
                return clause_count
            
            # Count clauses in all pages
            pages = DocumentValidator._get_pages_from_tiptap(document)
            for page in pages:
                page_content = page.get("content", [])
                for node in page_content:
                    if isinstance(node, dict):
                        count += count_clauses_in_node(node)
            
            # If no pages found, count from root content
            if not pages:
                content = document.get("content", [])
                for node in content:
                    if isinstance(node, dict):
                        count += count_clauses_in_node(node)
        else:
            # Legacy format: count clauses from sections
            pages = document.get("pages", [])
            for page in pages:
                sections = page.get("sections", [])
                for section in sections:
                    clauses = section.get("clauses", [])
                    count += len(clauses)
        
        return count

