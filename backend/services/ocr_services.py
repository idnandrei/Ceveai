import asyncio
import base64
import json
from typing import Any, Dict, List

import pymupdf
from dotenv import load_dotenv
from fastapi import UploadFile

from logger import app_log

from .llm_service import get_llm_service

# Load environment variables
load_dotenv()

# Below this many characters per page, a PDF is treated as scanned and sent to vision OCR
MIN_TEXT_CHARS_PER_PAGE = 100
# Render resolution for scanned pages sent to the vision model
OCR_DPI = 150

OCR_SYSTEM_PROMPT = """
You are an OCR assistant powered by a Vision-Language Model. Your job is to extract text and formatting information from any document, regardless of its format (images, PDFs, handwritten notes, etc.). You must output all extracted content in a well-organized Markdown document.
Key requirements:
Comprehensive Extraction: Capture every bit of text present in the document.
Structured Markdown Output: Organize the extracted text into a structured Markdown format.
Formatting Preservation:
Headers: Convert document headers into Markdown headers (using #, ##, etc.).
Footers: Identify and annotate footers appropriately.
Tables: Recognize tables and render them using Markdown table syntax.
Additional Features: Include lists, bold or italic text, page breaks, and any other formatting cues that can be represented in Markdown.
Detail-Oriented: Ensure that nothing is omitted—extract and present all available information from the document.
Your final output should be a single, structured Markdown document that faithfully represents both the content and the formatting of the original input.
"""


def _extract_pdf(pdf_bytes: bytes) -> tuple[str, List[bytes]]:
    """
    Extract the PDF's embedded text. If it has too little text (a scanned PDF),
    also render its pages to PNG so they can be OCR'd.
    Returns (text, page_images); page_images is empty when the text is usable.
    """
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        text = "\n\n".join(page.get_text() for page in doc).strip()
        if len(text) >= MIN_TEXT_CHARS_PER_PAGE * max(doc.page_count, 1):
            return text, []
        images = [page.get_pixmap(dpi=OCR_DPI).tobytes("png") for page in doc]
        return text, images


class OCRService:
    def __init__(self):
        self.llm_service = get_llm_service()

    async def _ocr_images(self, images: List[bytes]) -> str:
        """OCR all pages of a scanned document in a single vision call"""
        content = [
            {
                "type": "text",
                "text": "Please extract all text and formatting from these pages and present it as a well-structured Markdown document.",
            }
        ]
        for image in images:
            image_base64 = base64.b64encode(image).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                }
            )
        messages = [
            {"role": "system", "content": OCR_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        return await self.llm_service.generate_vision(messages)

    async def parse_document(self, document_file: UploadFile) -> Dict[str, Any]:
        """Parse document (PDF only) into structured data"""
        content_type = document_file.content_type or ""
        file_extension = (
            document_file.filename.split(".")[-1].lower()
            if document_file.filename
            else ""
        )

        # Check if it's a PDF
        if not (file_extension == "pdf" or "pdf" in content_type.lower()):
            return {
                "error": "Only PDF files are supported.",
                "document_type": file_extension.upper() if file_extension else "Unknown",
                "items": [],
            }

        try:
            await document_file.seek(0)
            pdf_bytes = await document_file.read()
            text, images = await asyncio.to_thread(_extract_pdf, pdf_bytes)

            if images:
                app_log.info(
                    f"{document_file.filename}: no text layer, OCR'ing {len(images)} pages"
                )
                text = await self._ocr_images(images)
            else:
                app_log.info(f"{document_file.filename}: extracted text directly")

            return {"document_type": "PDF", "markdown_content": text, "items": []}

        except Exception as e:
            app_log.error(f"Error parsing {document_file.filename}: {str(e)}")
            return {
                "document_type": "PDF",
                "markdown_content": "",
                "items": [],
                "error": str(e),
            }

    async def _analyze_cv(
        self, cv: dict, criteria: List[dict], job_description: str
    ) -> dict:
        # Build criteria string for the prompt
        criteria_str = "\n".join(
            f"- {c['name']}: {c['description']}" for c in criteria
        )

        # System prompt
        system_prompt = f"""
You are an expert CV analyzer with a focus on objective, data-driven evaluation. Your task is to analyze a CV and score it against specific criteria, always considering the job description when relevant.

Job Description:
{job_description}

Criteria (with descriptions):
{criteria_str}

Evaluation Guidelines:
1. Scoring System:
   - Use a 0-10 scale with one decimal point precision
   - 0: Completely missing or irrelevant
   - 5: Meets basic requirements
   - 10: Exceeds expectations significantly
   - When assigning scores, consider the full range of decimal values (e.g., 7.2, 8.3, 6.7, etc.) to best reflect nuanced differences in candidate performance.

2. Objectivity Requirements:
   - Base scores on concrete evidence from the CV
   - Avoid subjective interpretations
   - Consider quantifiable metrics when available

3. Job Description Integration:
   - For job-relevant criteria, explicitly map CV content to job requirements
   - For non-job-specific criteria (e.g., grammar), maintain objective standards

4. Response Format:
   - Each criterion must have a score (0-10 with one decimal point) and a one-sentence explanation
   - The summary should highlight key qualifications in about 100 words, as a single paragraph
   - Maintain the exact JSON structure provided in the user prompt
"""

        # User prompt
        user_prompt = f"""CV Content:
{cv['content']}

First, extract the candidate's full name from the CV. If the name is not specified, use 'N/A'. Then provide your analysis as JSON in the following format, using the exact criterion names as keys:
{{
    "candidate": "<full name from CV or N/A>",
    "scores": {{
        "<criterion name>": {{
            "score": <number with one decimal point>,
            "explanation": "<one-sentence reasoning for the score>"
        }}
    }},
    "summary": "<~100 word summary of key qualifications>"
}}
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = await self.llm_service.generate_response(messages, json_mode=True)
        try:
            result = json.loads(response)
        except Exception as e:
            result = {
                "error": f"Failed to parse LLM response: {str(e)}",
                "raw_response": response,
            }
        return {"filename": cv["filename"], **result}

    async def analyze_cvs(
        self,
        cv_contents: List[dict],  # Each dict: { "filename": ..., "content": ... }
        criteria: List[dict],  # Each dict: { "name": ..., "description": ... }
        job_description: str,
    ) -> List[dict]:
        """
        For each CV, get scores for each criterion and a summary of the CV.
        CVs are analyzed concurrently.
        Returns a list of dicts: { "filename": ..., "scores": ..., "summary": ... }
        """
        return await asyncio.gather(
            *(self._analyze_cv(cv, criteria, job_description) for cv in cv_contents)
        )
