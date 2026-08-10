"""Prompt builder for document generation."""

from typing import List, Dict, Any
from app.config import settings
from app.services.generation.language import language_label, normalize_language_code
import logging

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Service for constructing prompts for document generation."""

    DOCUMENT_STRUCTURE_EXAMPLE_AMHARIC = """
{
  "type": "doc",
  "content": [
    {
      "type": "page",
      "content": [
        {
          "type": "heading",
          "attrs": { "level": 1 },
          "content": [
            {
              "type": "text",
              "text": "የአጋርነት ስምምነት"
            }
          ]
        },
        {
          "type": "paragraph",
          "attrs": { "block_id": "b1" },
          "content": [
            {
              "type": "text",
              "text": "ይህ ስምምነት በዲሴምበር 1፣ 2025 የተፈጠረ ሲሆን በዴላዌር ግዛት ህጎች መሠረት የተዋቀረ ኩባንያ ኤ እና በካሊፎርኒያ ግዛት ህጎች መሠረት የተዋቀረ ኩባንያ ቢ መካከል ነው።"
            }
          ]
        },
        {
          "type": "heading",
          "attrs": { "level": 2 },
          "content": [
            {
              "type": "text",
              "text": "1. ዓላማ እና ወሰን"
            }
          ]
        },
        {
          "type": "paragraph",
          "attrs": { "block_id": "b2" },
          "content": [
            {
              "type": "text",
              "text": "የዚህ ስምምነት ዓላማ ሁለቱንም አጋሮች በአዲስ ምርቶች እና አገልግሎቶች ልማት፣ ገበያ እና ስርጭት ውስጥ በብቃት ለመተባበር የሚያስችላቸውን ማዕቀፍ ማቋቋም ነው።"
            }
          ]
        }
      ]
    },
    {
      "type": "pageBreak"
    },
    {
      "type": "page",
      "content": [
        {
          "type": "heading",
          "attrs": { "level": 2 },
          "content": [
            {
              "type": "text",
              "text": "2. ግዴታዎች እና ኃላፊነቶች"
            }
          ]
        },
        {
          "type": "paragraph",
          "attrs": { "block_id": "b3" },
          "content": [
            {
              "type": "text",
              "text": "እያንዳንዱ አጋር የተወሰኑ ግዴታዎች እና ኃላፊነቶች ይኖሩታል።"
            }
          ]
        }
      ]
    }
  ]
}
"""

    DOCUMENT_STRUCTURE_EXAMPLE_ENGLISH = """
{
  "type": "doc",
  "content": [
    {
      "type": "page",
      "content": [
        {
          "type": "heading",
          "attrs": { "level": 1 },
          "content": [
            {
              "type": "text",
              "text": "Partnership Agreement"
            }
          ]
        },
        {
          "type": "paragraph",
          "attrs": { "block_id": "b1" },
          "content": [
            {
              "type": "text",
              "text": "This agreement is entered into on December 1, 2025, between Company A, organized under Delaware law, and Company B, organized under California law."
            }
          ]
        },
        {
          "type": "heading",
          "attrs": { "level": 2 },
          "content": [
            {
              "type": "text",
              "text": "1. Purpose and Scope"
            }
          ]
        },
        {
          "type": "paragraph",
          "attrs": { "block_id": "b2" },
          "content": [
            {
              "type": "text",
              "text": "The purpose of this agreement is to establish a framework that enables both parties to collaborate effectively in the development, marketing, and distribution of innovative products and services."
            }
          ]
        }
      ]
    },
    {
      "type": "pageBreak"
    },
    {
      "type": "page",
      "content": [
        {
          "type": "heading",
          "attrs": { "level": 2 },
          "content": [
            {
              "type": "text",
              "text": "2. Obligations and Responsibilities"
            }
          ]
        },
        {
          "type": "paragraph",
          "attrs": { "block_id": "b3" },
          "content": [
            {
              "type": "text",
              "text": "Each party shall have specific obligations and responsibilities as outlined in this section."
            }
          ]
        }
      ]
    }
  ]
}
"""

    def build_analysis_prompt(self, user_prompt: str, conversation_history: List[Dict[str, Any]], context: Dict[str, Any] = None) -> str:
        """Build prompt for analyzing user requirements."""
        history_text = ""
        if conversation_history:
            history_text = "\n\nConversation History:\n"
            for msg in conversation_history[-5:]:  # Last 5 messages
                if msg.get("role") == "user":
                    history_text += f"User: {msg.get('content', '')}\n"
                elif msg.get("role") == "assistant":
                    history_text += f"Assistant: {msg.get('content', '')}\n"

        # Extract language from context or default to English
        language = "en"
        if context and context.get("language"):
            language = normalize_language_code(context.get("language"))
        
        # Language-specific instructions
        if language == "am":
            language_instruction = """
CRITICAL LANGUAGE REQUIREMENT:
- You MUST respond in Amharic (አማርኛ)
- All your questions, messages, and responses MUST be in Amharic
- Use Amharic text for all communication with the user
- The language field in your response MUST be "am"
- IMPORTANT: Do NOT translate the user's input - preserve it exactly as they wrote it
- The user may write in English or any language - keep their original text unchanged
"""
        elif language == "om":
            language_instruction = """
CRITICAL LANGUAGE REQUIREMENT:
- You MUST respond in Afaan Oromo (Oromo)
- All your questions, messages, and responses MUST be in Afaan Oromo
- The language field in your response MUST be "om"
- IMPORTANT: Do NOT translate the user's input - preserve it exactly as they wrote it
- The user may write in English or any language - keep their original text unchanged
"""
        else:
            language_instruction = """
CRITICAL LANGUAGE REQUIREMENT:
- You MUST respond in English
- All your questions, messages, and responses MUST be in English
- The language field in your response MUST be "en"
- IMPORTANT: Do NOT translate the user's input - preserve it exactly as they wrote it
- The user may write in any language - keep their original text unchanged
"""

        json_requirements = self._get_json_formatting_requirements()
        
        prompt = f"""You are a legal document generation assistant. Analyze the user's request and determine what information is needed.

{language_instruction}

User Request: {user_prompt}
{history_text}

IMPORTANT DEFAULT BEHAVIOR:
- By default, you MUST ask questions to gather complete information
- Only skip questions if the user EXPLICITLY says one of these phrases:
  * "keep it simple"
  * "use your own data"
  * "use defaults"
  * "use sample data"
  * "just generate it"
  * "generate a sample"
  * "I don't care about details"
  * Similar explicit requests to skip questions

ANALYSIS PROCESS:
1. Extract all available information from the user's request
2. Identify what critical information is missing (parties, dates, terms, amounts, etc.)
3. **CRITICAL: Extract document length/page count requirements:**
   - Look for phrases like: "X pages", "under X pages", "maximum X pages", "at least X pages", "keep it short", "brief", "concise", "comprehensive", "detailed"
   - If user says "under 3 pages" or "keep it short under 3 pages", extract num_pages: 3
   - If user says "4 pages", extract num_pages: 4
   - If user says "keep it short" or "brief" without a number, infer a shorter document (2-3 pages)
   - If user says "comprehensive" or "detailed", infer a longer document (5-6 pages)
   - Store the extracted page count in extracted_info as "num_pages": <integer>
4. Determine if the user explicitly requested to skip questions (see phrases above)
5. If NO explicit "keep it simple" request:
   - Set ready_to_generate to FALSE
   - Ask relevant questions in missing_info array
   - **If num_pages is NOT specified, add a question about desired document length:**
     * English: "How many pages should the document be? (e.g., 2-3 pages for a brief document, 4-6 pages for a standard document)"
     * Amharic: "መልእክቱ ስንት ገጾች ሊኖረው ይገባል? (ለአጭር ሰነድ 2-3 ገጾች፣ ለመደበኛ ሰነድ 4-6 ገጾች)"
   - Provide a friendly response asking for the missing information
6. If user EXPLICITLY requested simplicity:
   - Extract document type and language
   - Extract page count if mentioned, otherwise use default (4 pages)
   - Use GENERIC/SAMPLE data for missing information
   - Set ready_to_generate to TRUE
   - Set missing_info to empty array []
   - Do NOT ask questions

CRITICAL: Do NOT assume the user wants simplicity. Only treat as simple if they explicitly say so.

QUESTION FORMATTING REQUIREMENTS:
- Keep ALL questions SHORT and CONCISE
- Each question MUST be 1-2 lines maximum
- Use direct, simple language
- Avoid lengthy explanations or examples in questions
- Examples of good short questions:
  * "What type of legal agreement do you need?"
  * "Who are the parties and their addresses?"
  * "What is the effective date?"
  * "What is the monetary value and currency?"
  * "What jurisdiction or governing law applies?"
  * "How many pages should the document be?"
- DO NOT write long questions like "What is the specific type of legal agreement you need (e.g., Non-Disclosure Agreement, Sales Agreement, Service Agreement)?"

{json_requirements}

Respond with a JSON object containing:
- "document_type": string (e.g., "service agreement", "contract", "NDA")
- "language": string ("am" for Amharic, "en" for English, "om" for Afaan Oromo)
- "extracted_info": object with extracted info including "num_pages" if specified (only use defaults if user explicitly requested simplicity)
- "questions": array of questions (MUST include questions unless user explicitly said to skip) - KEEP EACH QUESTION TO 1-2 LINES MAX
- "ready_to_generate": boolean (true ONLY if you have complete info OR user explicitly requested simplicity)
- "response_message": string (friendly message, include questions if ready_to_generate is false)
"""
        return prompt

    def build_clarification_prompt(self, user_message: str, conversation_history: List[Dict[str, Any]], language: str = "en") -> str:
        """Build prompt for processing user clarification."""
        history_text = ""
        if conversation_history:
            history_text = "\n\nConversation History:\n"
            for msg in conversation_history[-10:]:  # Last 10 messages
                if msg.get("role") == "user":
                    history_text += f"User: {msg.get('content', '')}\n"
                elif msg.get("role") == "assistant":
                    history_text += f"Assistant: {msg.get('content', '')}\n"

        # Language-specific instructions
        lang = normalize_language_code(language) if language else "en"
        if lang == "am":
            language_instruction = """
CRITICAL LANGUAGE REQUIREMENT:
- You MUST respond in Amharic (አማርኛ)
- All your questions, messages, and responses MUST be in Amharic
- Use Amharic text for all communication with the user
- IMPORTANT: Do NOT translate the user's input - preserve it exactly as they wrote it
- The user may write in English or any language - keep their original text unchanged
"""
        elif lang == "om":
            language_instruction = """
CRITICAL LANGUAGE REQUIREMENT:
- You MUST respond in Afaan Oromo (Oromo)
- All your questions, messages, and responses MUST be in Afaan Oromo
- IMPORTANT: Do NOT translate the user's input - preserve it exactly as they wrote it
- The user may write in English or any language - keep their original text unchanged
"""
        else:
            language_instruction = """
CRITICAL LANGUAGE REQUIREMENT:
- You MUST respond in English
- All your questions, messages, and responses MUST be in English
- IMPORTANT: Do NOT translate the user's input - preserve it exactly as they wrote it
- The user may write in any language - keep their original text unchanged
"""

        json_requirements = self._get_json_formatting_requirements()
        
        prompt = f"""You are a legal document generation assistant. Process the user's response and update requirements.

{language_instruction}

User Response: {user_message}
{history_text}

IMPORTANT DEFAULT BEHAVIOR:
- By default, you MUST ask follow-up questions if information is still missing
- Only skip questions if the user EXPLICITLY says one of these phrases:
  * "keep it simple"
  * "use your own data"
  * "use defaults"
  * "use sample data"
  * "just generate it"
  * "that's enough"
  * "proceed"
  * "go ahead"
  * Similar explicit requests to proceed without more questions

PROCESSING:
1. Extract any new information from the user's response
2. **CRITICAL: Extract document length/page count requirements:**
   - Look for phrases like: "X pages", "under X pages", "maximum X pages", "at least X pages", "keep it short", "brief", "concise", "comprehensive", "detailed"
   - If user says "under 3 pages" or "keep it short under 3 pages", extract num_pages: 3
   - If user says "4 pages", extract num_pages: 4
   - If user says "keep it short" or "brief" without a number, infer a shorter document (2-3 pages)
   - If user says "comprehensive" or "detailed", infer a longer document (5-6 pages)
   - Update updated_requirements with "num_pages": <integer>
3. Update requirements with the new information
4. Determine what critical information is still missing
5. Check if the user EXPLICITLY requested to skip questions (see phrases above)
6. If NO explicit "proceed" request AND information is missing:
   - Set ready_to_generate to FALSE
   - Ask follow-up questions in questions array
   - **If num_pages is still NOT specified, add a question about desired document length:**
     * English: "How many pages should the document be? (e.g., 2-3 pages for a brief document, 4-6 pages for a standard document)"
     * Amharic: "መልእክቱ ስንት ገጾች ሊኖረው ይገባል? (ለአጭር ሰነድ 2-3 ገጾች፣ ለመደበኛ ሰነድ 4-6 ገጾች)"
   - Provide a friendly response asking for remaining information
7. If user EXPLICITLY requested to proceed OR you have complete information:
   - Use GENERIC/SAMPLE data for any remaining missing information (if user requested simplicity)
   - If num_pages is still not specified, use default (4 pages)
   - Set ready_to_generate to TRUE
   - Set questions to empty array []
   - Do NOT ask more questions

CRITICAL: Do NOT assume the user wants to proceed. Only treat as ready if you have complete info OR they explicitly said to proceed.

QUESTION FORMATTING REQUIREMENTS:
- Keep ALL questions SHORT and CONCISE
- Each question MUST be 1-2 lines maximum
- Use direct, simple language
- Avoid lengthy explanations or examples in questions
- Examples of good short questions:
  * "What type of legal agreement do you need?"
  * "Who are the parties and their addresses?"
  * "What is the effective date?"
  * "What is the monetary value and currency?"
  * "What jurisdiction or governing law applies?"
  * "How many pages should the document be?"
- DO NOT write long questions with examples or lengthy explanations

{json_requirements}

Respond with a JSON object containing:
- "updated_requirements": object with updated/extracted information including "num_pages" if specified
- "ready_to_generate": boolean (true ONLY if you have complete info OR user explicitly said to proceed)
- "response_message": string (acknowledge input, include questions if ready_to_generate is false)
- "questions": array of strings (MUST include questions if information is missing and user didn't explicitly say to proceed) - KEEP EACH QUESTION TO 1-2 LINES MAX
"""
        return prompt

    def build_generation_prompt(
        self,
        requirements: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
        knowledge_context: str = "",
    ) -> str:
        """Build prompt for generating the document structure."""
        history_text = ""
        if conversation_history:
            history_text = "\n\nRecent Conversation Context:\n"
            # Limit to last 3 messages to reduce prompt size
            for msg in conversation_history[-3:]:
                content = msg.get('content', '')
                # Truncate very long messages to 300 chars
                if len(content) > 300:
                    content = content[:300] + "..."
                if msg.get("role") == "user":
                    history_text += f"User: {content}\n"
                elif msg.get("role") == "assistant":
                    history_text += f"Assistant: {content}\n"

        # Select appropriate example based on language (normalize aliases like amh/oro)
        language = normalize_language_code(requirements.get("language", "en"))
        if language == "am":
            structure_example = self.DOCUMENT_STRUCTURE_EXAMPLE_AMHARIC
            language_instruction = """
CRITICAL LANGUAGE REQUIREMENT:
- You MUST generate the ENTIRE document in Amharic (አማርኛ)
- All text, titles, content, clauses MUST be in Amharic
- Use Amharic characters and proper Amharic formatting
- The metadata language field MUST be "am"
- IMPORTANT: Preserve user-provided information (names, addresses, dates, etc.) exactly as provided
- If the user provided names/addresses in English, keep them in English within the Amharic document
- Only translate your generated legal text to Amharic, not the user's specific information
"""
            final_language_reminder = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL LANGUAGE CHECK (OVERRIDES PLACEHOLDER EXAMPLES ABOVE):
- The English text in the JSON syntax examples is PLACEHOLDER ONLY for comma/brace rules.
- Your actual document body, headings, and clauses MUST be written in Amharic (አማርኛ).
- metadata.language MUST be "am".
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        elif language == "om":
            structure_example = self.DOCUMENT_STRUCTURE_EXAMPLE_ENGLISH
            language_instruction = """
CRITICAL LANGUAGE REQUIREMENT:
- You MUST generate the ENTIRE document in Afaan Oromo (Oromo)
- All text, titles, content, clauses MUST be in Afaan Oromo
- The metadata language field MUST be "om"
- IMPORTANT: Preserve user-provided information (names, addresses, dates, etc.) exactly as provided
- Only translate your generated legal text to Afaan Oromo, not the user's specific information
"""
            final_language_reminder = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL LANGUAGE CHECK (OVERRIDES PLACEHOLDER EXAMPLES ABOVE):
- The English text in the JSON syntax examples is PLACEHOLDER ONLY for comma/brace rules.
- Your actual document body, headings, and clauses MUST be written in Afaan Oromo.
- metadata.language MUST be "om".
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        else:
            structure_example = self.DOCUMENT_STRUCTURE_EXAMPLE_ENGLISH
            language_instruction = """
CRITICAL LANGUAGE REQUIREMENT:
- You MUST generate the ENTIRE document in English
- All text, titles, content, clauses MUST be in English
- The metadata language field MUST be "en"
- IMPORTANT: Preserve user-provided information (names, addresses, dates, etc.) exactly as provided
- Keep all user-provided information in its original language
"""
            final_language_reminder = ""

        # Extract document type for compliance requirements
        document_type = requirements.get("document_type", "Contract")
        if not document_type or document_type == "":
            # Try to infer from other fields
            doc_type_lower = str(requirements.get("template_type", "")).lower()
            if "employment" in doc_type_lower or "employee" in doc_type_lower:
                document_type = "Employment Contract"
            elif "partnership" in doc_type_lower:
                document_type = "Partnership Agreement"
            elif "nda" in doc_type_lower or "non-disclosure" in doc_type_lower:
                document_type = "NDA"
            elif "mou" in doc_type_lower or "memorandum" in doc_type_lower:
                document_type = "MOU"
            elif "power of attorney" in doc_type_lower or "poa" in doc_type_lower:
                document_type = "Power of Attorney"
            else:
                document_type = "Contract"

        # Build compliance requirements section
        compliance_requirements = self._build_compliance_requirements(document_type, language)

        # Include knowledge context if provided
        knowledge_section = ""
        if knowledge_context:
            knowledge_section = f"\n\n{knowledge_context}\n"

        json_requirements = self._get_json_formatting_requirements()

        # Get user-specified page count or use default
        target_pages = requirements.get("num_pages")
        if target_pages and isinstance(target_pages, int) and 1 <= target_pages <= 20:
            # User specified a valid page count
            min_pages = target_pages
            max_pages = min(target_pages + 1, settings.MAX_PAGES)  # Allow 1 page buffer, but cap at max
        else:
            # Use default settings
            min_pages = settings.MIN_PAGES
            max_pages = settings.MAX_PAGES

        prompt = f"""You are a legal document generation assistant. Generate a complete, comprehensive, and legally compliant document structure based on the requirements.

{language_instruction}

{compliance_requirements}

🚨 CRITICAL LENGTH REQUIREMENTS - MUST BE MET IN THIS SINGLE GENERATION 🚨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOU MUST GENERATE A COMPLETE DOCUMENT RIGHT NOW WITH:
✓ TARGET: {min_pages} pages (REQUIRED - generate exactly this many pages)
✓ MAXIMUM: {max_pages} pages (HARD LIMIT - DO NOT EXCEED THIS)
✓ EACH PAGE: {settings.MIN_WORDS_PER_PAGE}+ words of detailed, comprehensive content

⚠️  CRITICAL: If you generate MORE than {max_pages} pages, the document will be TRUNCATED
⚠️  DO NOT generate a short or incomplete document
⚠️  DO NOT expect expansion or regeneration later  
⚠️  THIS IS YOUR ONLY CHANCE - make it complete and comprehensive NOW
⚠️  DO NOT exceed {max_pages} pages - this is a HARD LIMIT
⚠️  AIM FOR EXACTLY {min_pages} PAGES - this is what the user requested

If you generate less than {min_pages} pages or fewer than {settings.MIN_WORDS_PER_PAGE} words per page,
the document will FAIL validation and the user will have to wait for regeneration.

If you generate MORE than {max_pages} pages, the document will be automatically truncated to {max_pages} pages.

BE VERBOSE. BE DETAILED. BE COMPREHENSIVE. Generate exactly {min_pages} pages (or up to {max_pages} if needed for completeness).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Requirements:
{self._format_requirements(requirements)}
{history_text}
{knowledge_section}
Generate a complete document matching this EXACT structure:
{structure_example}

IMPORTANT TECHNICAL REQUIREMENTS:
1. Generate valid TipTap JSON format only
2. Match the EXACT structure shown above
3. Document must have "type": "doc" at root
4. Use "page" nodes separated by "pageBreak" nodes
5. Every paragraph MUST have "attrs": {{"block_id": "bX"}} where X is a unique number
6. Use "heading" nodes with "attrs": {{"level": 1-6}} for section titles
7. Use "paragraph" nodes for text content
8. Use "bulletList" and "listItem" for lists when appropriate
9. All text must be in "text" nodes with "type": "text" and "text": "content"
10. Include all standard legal clauses and provisions relevant to the document type
11. Use the legal knowledge provided above to ensure compliance and accuracy
12. Make the document comprehensive, detailed, and legally sound

TIPTAP STRUCTURE RULES:
- Root: {{"type": "doc", "content": [...]}}
- Page: {{"type": "page", "content": [...]}}
- PageBreak: {{"type": "pageBreak"}}
- Heading: {{"type": "heading", "attrs": {{"level": N}}, "content": [...]}}
- Paragraph: {{"type": "paragraph", "attrs": {{"block_id": "bN"}}, "content": [...]}}
- Text: {{"type": "text", "text": "your text here"}}
- Assign sequential block_ids: b1, b2, b3, etc.

CONTENT REQUIREMENTS:
- Each page should be rich with detailed legal provisions
- Include comprehensive definitions, obligations, rights, and responsibilities
- Add standard clauses: termination, dispute resolution, governing law, etc.
- Ensure all content is legally accurate and compliant with Ethiopian legal framework
- Use proper Ethiopian legal terminology and phrasing
- CRITICAL: All clauses MUST comply with the compliance requirements specified above
- Reference the compliance requirements section when drafting each clause
- Use the legal knowledge provided above to ensure compliance and accuracy

{json_requirements}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📚 FEW-SHOT EXAMPLES - CORRECT JSON WITH PROPER COMMA PLACEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE 1 - Correct comma placement between list items:
{{
  "type": "doc",
  "content": [
    {{
      "type": "page",
      "content": [
        {{
          "type": "paragraph",
          "attrs": {{"block_id": "b1"}},
          "content": [{{"type": "text", "text": "First paragraph"}}]
        }},
        {{
          "type": "paragraph",
          "attrs": {{"block_id": "b2"}},
          "content": [{{"type": "text", "text": "Second paragraph"}}]
        }}
      ]
    }}
  ]
}}

✓ NOTICE: Comma after }} before next object
✓ NOTICE: Comma after }} before closing ]

EXAMPLE 2 - Correct comma placement in nested structures:
{{
  "type": "listItem",
  "content": [
    {{
      "type": "paragraph",
      "attrs": {{"block_id": "b10"}},
      "content": [{{"type": "text", "text": "Item text"}}]
    }}
  ]
}},
{{
  "type": "listItem",
  "content": [
    {{
      "type": "paragraph",
      "attrs": {{"block_id": "b11"}},
      "content": [{{"type": "text", "text": "Another item"}}]
    }}
  ]
}}

✓ NOTICE: Comma after }} before next listItem
✓ NOTICE: Comma after ] before next object

EXAMPLE 3 - WRONG (missing comma - will cause parsing error):
{{
  "type": "paragraph",
  "attrs": {{"block_id": "b1"}},
  "content": [{{"type": "text", "text": "Text"}}]
}}
{{
  "type": "paragraph",
  "attrs": {{"block_id": "b2"}},
  "content": [{{"type": "text", "text": "More text"}}]
}}

❌ MISSING COMMA after first }} - this will FAIL parsing!

EXAMPLE 4 - CORRECT (comma added):
{{
  "type": "paragraph",
  "attrs": {{"block_id": "b1"}},
  "content": [{{"type": "text", "text": "Text"}}]
}},
{{
  "type": "paragraph",
  "attrs": {{"block_id": "b2"}},
  "content": [{{"type": "text", "text": "More text"}}]
}}

✓ COMMA PRESENT after first }} - this will parse correctly!

EXAMPLE 5 - WRONG (missing comma after closing brace in array):
{{
  "type": "listItem",
  "content": [
    {{
      "type": "paragraph",
      "attrs": {{"block_id": "b10"}},
      "content": [{{"type": "text", "text": "Item 1"}}]
    }}
  ]
}}
{{
  "type": "listItem",
  "content": [
    {{
      "type": "paragraph",
      "attrs": {{"block_id": "b11"}},
      "content": [{{"type": "text", "text": "Item 2"}}]
    }}
  ]
}}

❌ MISSING COMMA after first }} - this will cause "Expecting ',' delimiter" error!

EXAMPLE 6 - CORRECT (comma added after closing brace):
{{
  "type": "listItem",
  "content": [
    {{
      "type": "paragraph",
      "attrs": {{"block_id": "b10"}},
      "content": [{{"type": "text", "text": "Item 1"}}]
    }}
  ]
}},
{{
  "type": "listItem",
  "content": [
    {{
      "type": "paragraph",
      "attrs": {{"block_id": "b11"}},
      "content": [{{"type": "text", "text": "Item 2"}}]
    }}
  ]
}}

✓ COMMA PRESENT after first }} - this will parse correctly!

EXAMPLE 7 - Complete valid TipTap document structure:
{{
  "type": "doc",
  "content": [
    {{
      "type": "page",
      "content": [
        {{
          "type": "heading",
          "attrs": {{"level": 1}},
          "content": [{{"type": "text", "text": "Document Title"}}]
        }},
        {{
          "type": "paragraph",
          "attrs": {{"block_id": "b1"}},
          "content": [{{"type": "text", "text": "First paragraph text."}}]
        }},
        {{
          "type": "paragraph",
          "attrs": {{"block_id": "b2"}},
          "content": [{{"type": "text", "text": "Second paragraph text."}}]
        }}
      ]
    }},
    {{
      "type": "pageBreak"
    }},
    {{
      "type": "page",
      "content": [
        {{
          "type": "heading",
          "attrs": {{"level": 2}},
          "content": [{{"type": "text", "text": "Section Title"}}]
        }},
        {{
          "type": "paragraph",
          "attrs": {{"block_id": "b3"}},
          "content": [{{"type": "text", "text": "More content."}}]
        }}
      ]
    }}
  ]
}}

✓ NOTICE: Every }} or ]] that is NOT the last element has a comma after it
✓ NOTICE: Commas separate all array elements and object properties
✓ NOTICE: No trailing commas before }} or ]]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL REMINDER: After EVERY }} or ]] that is NOT the last element, add a comma!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Respond with ONLY the JSON document structure, no additional text. Start with {{ and end with }} - pure JSON only.
"""
        return prompt + final_language_reminder

    def _format_requirements(self, requirements: Dict[str, Any]) -> str:
        """Format requirements for prompt."""
        formatted = []
        for key, value in requirements.items():
            if key == "language":
                code = normalize_language_code(value)
                formatted.append(
                    f"language: {code} — OUTPUT LANGUAGE: {language_label(code)} "
                    f"(generate the entire document in this language; "
                    f'metadata.language must be "{code}")'
                )
                continue
            if isinstance(value, dict):
                formatted.append(f"{key}:")
                for k, v in value.items():
                    formatted.append(f"  {k}: {v}")
            elif isinstance(value, list):
                formatted.append(f"{key}: {', '.join(str(v) for v in value)}")
            else:
                formatted.append(f"{key}: {value}")
        return "\n".join(formatted)

    def _get_document_type_compliance_rules(self, document_type: str) -> Dict[str, Any]:
        """Get compliance rules for a specific document type."""
        doc_type_lower = document_type.lower() if document_type else ""
        
        # Employment Contract specific rules
        if "employment" in doc_type_lower or "employee" in doc_type_lower or "employer" in doc_type_lower:
            return {
                "critical_prohibitions": [
                    "NO 'at-will' employment clauses - these are ILLEGAL under Ethiopian Labor Law (Proclamation No. 1156/2019)",
                    "NO termination without cause provisions - must follow proper termination procedures",
                    "NO foreign governing law - MUST use Ethiopian law",
                    "NO foreign jurisdiction clauses - MUST use Ethiopian courts",
                ],
                "mandatory_clauses": [
                    "Adequate notice period for resignation (minimum 30 days as per Ethiopian Labor Law)",
                    "Proper termination procedures compliant with Ethiopian Labor Law",
                    "Mandatory benefits: social security contributions, annual leave, sick leave, maternity/paternity leave",
                    "Employer tax obligations under Ethiopian tax law",
                    "Governing law: Ethiopian law (Proclamation No. 1156/2019 and related legislation)",
                    "Jurisdiction: Ethiopian courts for dispute resolution",
                    "Working hours and overtime provisions compliant with Ethiopian Labor Law",
                    "Salary payment terms and currency (Ethiopian Birr)",
                ],
                "compliance_focus": "Ethiopian Labor Law (Proclamation No. 1156/2019) compliance is CRITICAL"
            }
        
        # Partnership Agreement specific rules
        elif "partnership" in doc_type_lower or "partner" in doc_type_lower:
            return {
                "critical_prohibitions": [
                    "NO foreign governing law - MUST use Ethiopian law",
                    "NO foreign jurisdiction clauses - MUST use Ethiopian courts",
                ],
                "mandatory_clauses": [
                    "Governing law: Ethiopian Commercial Code (1960) and Ethiopian law",
                    "Jurisdiction: Ethiopian courts for dispute resolution",
                    "Tax obligations under Ethiopian tax law",
                    "Partnership registration requirements under Ethiopian law",
                    "Profit and loss distribution terms",
                    "Partnership dissolution procedures",
                ],
                "compliance_focus": "Ethiopian Commercial Code (1960) compliance"
            }
        
        # NDA specific rules
        elif "nda" in doc_type_lower or "non-disclosure" in doc_type_lower or "confidentiality" in doc_type_lower:
            return {
                "critical_prohibitions": [
                    "NO foreign governing law - MUST use Ethiopian law",
                    "NO foreign jurisdiction clauses - MUST use Ethiopian courts",
                ],
                "mandatory_clauses": [
                    "Governing law: Ethiopian Civil Code (1960) and Ethiopian law",
                    "Jurisdiction: Ethiopian courts for dispute resolution",
                    "Definition of confidential information",
                    "Duration of confidentiality obligations",
                    "Exceptions to confidentiality",
                    "Remedies for breach",
                ],
                "compliance_focus": "Ethiopian Civil Code (1960) compliance"
            }
        
        # MOU specific rules
        elif "mou" in doc_type_lower or "memorandum" in doc_type_lower:
            return {
                "critical_prohibitions": [
                    "NO foreign governing law - MUST use Ethiopian law",
                    "NO foreign jurisdiction clauses - MUST use Ethiopian courts",
                ],
                "mandatory_clauses": [
                    "Governing law: Ethiopian law",
                    "Jurisdiction: Ethiopian courts for dispute resolution",
                    "Purpose and scope of understanding",
                    "Non-binding nature (if applicable)",
                    "Term and termination",
                ],
                "compliance_focus": "Ethiopian legal framework compliance"
            }
        
        # Power of Attorney specific rules
        elif "power of attorney" in doc_type_lower or "poa" in doc_type_lower:
            return {
                "critical_prohibitions": [
                    "NO foreign governing law - MUST use Ethiopian law",
                    "NO foreign jurisdiction clauses - MUST use Ethiopian courts",
                ],
                "mandatory_clauses": [
                    "Governing law: Ethiopian Civil Code (1960) and Ethiopian law",
                    "Jurisdiction: Ethiopian courts",
                    "Scope of authority granted",
                    "Duration and revocation terms",
                    "Notarization requirements (if applicable)",
                ],
                "compliance_focus": "Ethiopian Civil Code (1960) compliance"
            }
        
        # Default/General contract rules
        else:
            return {
                "critical_prohibitions": [
                    "NO foreign governing law - MUST use Ethiopian law",
                    "NO foreign jurisdiction clauses - MUST use Ethiopian courts",
                ],
                "mandatory_clauses": [
                    "Governing law: Ethiopian law (Ethiopian Commercial Code, Civil Code, or relevant legislation)",
                    "Jurisdiction: Ethiopian courts for dispute resolution",
                    "Dispute resolution mechanism (mediation, arbitration, or court proceedings under Ethiopian law)",
                    "Tax obligations under Ethiopian tax law (if applicable)",
                ],
                "compliance_focus": "Ethiopian legal framework compliance (Commercial Code, Civil Code, Constitution)"
            }

    def _build_compliance_requirements(self, document_type: str, language: str) -> str:
        """Build compliance requirements section for the generation prompt."""
        rules = self._get_document_type_compliance_rules(document_type)
        language = normalize_language_code(language)
        lang_name = language_label(language)
        
        # Build prohibitions section
        prohibitions_text = ""
        if rules.get("critical_prohibitions"):
            prohibitions_text = "\n".join([f"  ❌ {prohibition}" for prohibition in rules["critical_prohibitions"]])
        
        # Build mandatory clauses section
        mandatory_text = ""
        if rules.get("mandatory_clauses"):
            mandatory_text = "\n".join([f"  ✓ {clause}" for clause in rules["mandatory_clauses"]])
        
        compliance_section = f"""
🚨 CRITICAL ETHIOPIAN LEGAL COMPLIANCE REQUIREMENTS 🚨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DOCUMENT LANGUAGE: Write all clause text in {lang_name}.

ALL CLAUSES IN THIS DOCUMENT MUST COMPLY WITH ETHIOPIAN LEGAL FRAMEWORK:
- Ethiopian Commercial Code (1960)
- Ethiopian Civil Code (1960)
- Ethiopian Labor Law (Proclamation No. 1156/2019)
- Ethiopian Investment Law (Proclamation No. 1180/2020)
- Ethiopian Constitution (1995)

COMPLIANCE FOCUS: {rules.get("compliance_focus", "Ethiopian legal framework")}

CRITICAL PROHIBITIONS - DO NOT INCLUDE THESE (NON-COMPLIANT):
{prohibitions_text}

MANDATORY CLAUSES - MUST BE INCLUDED:
{mandatory_text}

GOVERNING LAW AND JURISDICTION (CRITICAL):
✓ Governing law clause MUST state: "This agreement shall be governed by and construed in accordance with the laws of Ethiopia"
✓ Jurisdiction clause MUST state: "Any disputes arising from this agreement shall be subject to the exclusive jurisdiction of Ethiopian courts"
✓ DO NOT use foreign law (US law, Delaware law, California law, etc.) - this will result in CRITICAL non-compliance
✓ DO NOT use foreign jurisdiction - this will result in CRITICAL non-compliance

GENERAL COMPLIANCE REQUIREMENTS:
✓ All clauses must align with Ethiopian legal standards and practices
✓ Use proper Ethiopian legal terminology in {lang_name}
✓ Reference relevant Ethiopian laws and proclamations where appropriate
✓ Ensure all provisions are enforceable under Ethiopian law
✓ Include all standard clauses required for this document type under Ethiopian law

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: Failure to comply with these requirements will result in a LOW compliance score.
The document will be flagged as non-compliant if it contains foreign law, foreign jurisdiction,
or missing mandatory clauses. Ensure ALL clauses comply with Ethiopian legal framework.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return compliance_section

    def _get_json_formatting_requirements(self) -> str:
        """Get comprehensive JSON formatting requirements for prompts."""
        return """
🚨 CRITICAL JSON FORMATTING REQUIREMENTS - FOLLOW THESE EXACTLY 🚨
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PROPERTY NAME QUOTES (CRITICAL - MOST COMMON ERROR):
✓ EVERY property name MUST have BOTH opening and closing double quotes
✓ CORRECT: "type": "doc", "content": [...], "attrs": {...}
✓ WRONG: type": "doc" (missing opening quote)
✓ WRONG: content": [...] (missing opening quote)
✓ WRONG: attrs": {...} (missing opening quote)
✓ Pattern: "property_name": value (opening quote + name + closing quote + colon)
✓ ALWAYS check: Every property name starts with " and ends with " before the colon

STRING VALUE QUOTES:
✓ ALL double quotes inside string values MUST be escaped with backslash: \\"
✓ Example: "text": "The \\"Party\\" agrees..." (quotes inside text are escaped)
✓ Example: "content": "<p>\\"text\\" with quotes</p>" (HTML quotes are escaped)

ESCAPE SEQUENCES:
✓ ALL newlines in strings MUST be escaped: \\n
✓ ALL tabs in strings MUST be escaped: \\t
✓ ALL backslashes MUST be escaped: \\\\
✓ Example: "text": "Line 1\\nLine 2\\tIndented"

COMMAS BETWEEN ELEMENTS (CRITICAL - PREVENTS PARSING ERRORS):
✓ EVERY element in arrays MUST be separated by commas
✓ EVERY key-value pair in objects MUST be separated by commas (except the last one)
✓ CORRECT: {"key1": "value1", "key2": "value2"}
✓ CORRECT: [{"type": "text"}, {"type": "paragraph"}]
✓ WRONG: } "key": (missing comma after })
✓ WRONG: ] { (missing comma after ])
✓ WRONG: "value" "key": (missing comma between string values)

🎯 VISUAL EXAMPLES - COMMA PLACEMENT:

CORRECT Pattern 1 - Comma after closing brace in array:
[
  {{"type": "paragraph", "attrs": {{"block_id": "b1"}}}},
  {{"type": "paragraph", "attrs": {{"block_id": "b2"}}}}
]
↑ Notice the comma after first }}

CORRECT Pattern 2 - Comma after closing bracket:
[
  {{"type": "listItem", "content": [...]}},
  {{"type": "listItem", "content": [...]}}
]
↑ Notice the comma after first ]}

WRONG Pattern 1 - Missing comma (WILL FAIL):
[
  {{"type": "paragraph", "attrs": {{"block_id": "b1"}}}}
  {{"type": "paragraph", "attrs": {{"block_id": "b2"}}}}
]
❌ Missing comma - parser will fail with "Expecting ',' delimiter"

WRONG Pattern 2 - Missing comma (WILL FAIL):
[
  {{"type": "listItem", "content": [...]}}
  {{"type": "listItem", "content": [...]}}
]
❌ Missing comma - parser will fail with "Expecting ',' delimiter"

✓ Pattern: After every } or ] that's not the last element, there MUST be a comma
✓ Check: If you see } followed by " or { or [, there MUST be a comma in between
✓ Check: If you see ] followed by " or { or [, there MUST be a comma in between
✓ Check: If you see "value" followed by "key":, there MUST be a comma in between

📋 SELF-VALIDATION CHECKLIST (verify before outputting):
□ Every } that is NOT the last element has a comma after it
□ Every ] that is NOT the last element has a comma after it
□ Every string value that is NOT the last element has a comma after it
□ No } or ] is directly followed by " or { or [ without a comma
□ All array elements are separated by commas
□ All object properties are separated by commas (except the last one)
□ No trailing commas before } or ]

STRUCTURE REQUIREMENTS:
✓ Return ONLY valid JSON - no markdown code blocks (```json or ```), no explanatory text, no comments
✓ No trailing commas before } or ] (e.g., "key": "value",} is WRONG, use "key": "value"})
✓ All strings use double quotes (never single quotes)
✓ All brackets [ ] and braces { } must be properly matched and closed
✓ No comments (// or /* */) anywhere in the JSON
✓ Every comma must be followed by a key, value, or closing bracket
✓ The JSON must be parseable by standard JSON parsers

VALIDATION CHECKLIST - VERIFY BEFORE OUTPUTTING:
□ Every property name has opening " and closing " before the colon
□ Every string value has opening " and closing "
□ All quotes inside string values are escaped as \\"
□ All newlines in strings are \\n
□ All tabs in strings are \\t
□ Commas between all array elements (check: } or ] followed by " or { or [ has comma)
□ Commas between all object key-value pairs (except the last one)
□ No trailing commas before } or ]
□ All brackets and braces are matched
□ No markdown formatting around JSON
□ Starts with { and ends with }

COMMON ERRORS TO AVOID:
❌ content": [...] → ✓ "content": [...]
❌ type": "doc" → ✓ "type": "doc"
❌ attrs": {...} → ✓ "attrs": {...}
❌ "text": "Unescaped "quote" here" → ✓ "text": "Escaped \\"quote\\" here"
❌ "key": "value",} → ✓ "key": "value"}
❌ } "key": → ✓ }, "key":
❌ ] { → ✓ ], {
❌ "value" "key": → ✓ "value", "key":

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: Double-check every property name has BOTH opening and closing quotes before outputting.
If you see a property name without an opening quote, you have made an error that will break the JSON parser.

CRITICAL: Double-check that ALL commas are present between elements.
If you see } or ] followed by " or { or [ without a comma, you have made an error that will break the JSON parser.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

