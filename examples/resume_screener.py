"""
Resume Screener Agent

This example demonstrates how to build a resume screening agent using:
- A custom PDFReaderWorker: For extracting text from PDF resumes
- HumanProxyWorker: For human-in-the-loop decisions
- Structured output: For consistent screening results

The agent can:
1. Read PDF resume files
2. Extract key information (skills, experience, education)
3. Score candidates against job requirements
4. Request human approval for borderline cases

Usage:
    python examples/resume_screener.py resume.pdf
    
Requirements:
    pip install blackboard-core[stdlib] pypdf2
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from blackboard import Orchestrator, Blackboard, Status
from blackboard.protocols import Worker, WorkerInput, WorkerOutput
from blackboard.state import Artifact
from blackboard.llm import OpenAIClient


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class JobRequirements:
    """Defines job requirements for screening."""
    title: str
    required_skills: List[str]
    preferred_skills: List[str]
    min_experience_years: int
    education_level: str  # "bachelor", "master", "phd", "any"
    
    def to_prompt(self) -> str:
        return f"""
JOB REQUIREMENTS:
- Title: {self.title}
- Required Skills: {', '.join(self.required_skills)}
- Preferred Skills: {', '.join(self.preferred_skills)}
- Minimum Experience: {self.min_experience_years} years
- Education: {self.education_level}
"""


# =============================================================================
# Custom Workers
# =============================================================================

class PDFReaderWorker(Worker):
    """
    Reads and extracts text from PDF files.
    
    Input Schema:
        - file_path: Path to the PDF file
    """
    
    name = "PDFReader"
    description = "Extracts text content from PDF files (resumes, documents)"
    parallel_safe = True
    
    async def run(self, state: Blackboard, inputs: WorkerInput = None) -> WorkerOutput:
        # Get file path from state or inputs
        file_path = None
        if inputs and hasattr(inputs, 'instructions'):
            file_path = inputs.instructions.strip()
        if not file_path:
            file_path = state.metadata.get("resume_path")
        
        if not file_path or not Path(file_path).exists():
            return WorkerOutput(
                metadata={"error": f"PDF file not found: {file_path}"}
            )
        
        try:
            # Try using pypdf2
            try:
                from pypdf2 import PdfReader
            except ImportError:
                try:
                    from PyPDF2 import PdfReader
                except ImportError:
                    return WorkerOutput(
                        metadata={"error": "PyPDF2 not installed. Run: pip install pypdf2"}
                    )
            
            reader = PdfReader(file_path)
            text_content = []
            
            for page in reader.pages:
                text_content.append(page.extract_text() or "")
            
            full_text = "\n\n".join(text_content)
            
            return WorkerOutput(
                artifact=Artifact(
                    type="resume_text",
                    content=full_text,
                    creator=self.name,
                    metadata={
                        "file_path": str(file_path),
                        "page_count": len(reader.pages),
                        "char_count": len(full_text)
                    }
                )
            )
            
        except Exception as e:
            return WorkerOutput(
                metadata={"error": f"Failed to read PDF: {str(e)}"}
            )


class ResumeAnalyzer(Worker):
    """
    Analyzes resume content against job requirements.
    
    Uses LLM to extract structured information and score the candidate.
    """
    
    name = "ResumeAnalyzer"
    description = "Analyzes resume content and scores candidate against job requirements"
    parallel_safe = True
    
    def __init__(self, llm=None, requirements: Optional[JobRequirements] = None):
        self.llm = llm
        self.requirements = requirements or JobRequirements(
            title="Software Engineer",
            required_skills=["Python", "SQL", "Git"],
            preferred_skills=["AWS", "Docker", "Kubernetes"],
            min_experience_years=2,
            education_level="bachelor"
        )
    
    async def run(self, state: Blackboard, inputs: WorkerInput = None) -> WorkerOutput:
        # Find resume text in artifacts
        resume_artifact = None
        for artifact in state.artifacts:
            if artifact.type == "resume_text":
                resume_artifact = artifact
                break
        
        if not resume_artifact:
            return WorkerOutput(
                metadata={"error": "No resume text found. Run PDFReader first."}
            )
        
        prompt = f"""Analyze this resume against the job requirements and provide a structured assessment.

{self.requirements.to_prompt()}

RESUME CONTENT:
{resume_artifact.content[:4000]}

Provide your analysis in this exact JSON format:
{{
    "candidate_name": "Name from resume",
    "matched_required_skills": ["skill1", "skill2"],
    "matched_preferred_skills": ["skill1"],
    "missing_required_skills": ["skill1"],
    "years_of_experience": 3,
    "education": "Bachelor's in Computer Science",
    "overall_score": 75,
    "recommendation": "PROCEED" or "REVIEW" or "REJECT",
    "summary": "Brief 2-3 sentence summary"
}}

Be objective and base your assessment only on the resume content."""

        if self.llm:
            response = await self.llm.generate(prompt)
            content = response if isinstance(response, str) else response.content
        else:
            content = '{"error": "No LLM configured"}'
        
        return WorkerOutput(
            artifact=Artifact(
                type="screening_result",
                content=content,
                creator=self.name,
                metadata={
                    "job_title": self.requirements.title,
                    "source_artifact": resume_artifact.id
                }
            )
        )


# =============================================================================
# Main Application
# =============================================================================

def create_orchestrator(resume_path: str, requirements: Optional[JobRequirements] = None):
    """Factory function to create the resume screening orchestrator."""
    from blackboard.stdlib import CLIHumanProxyWorker
    
    # Initialize LLM
    llm = OpenAIClient(model="gpt-4o-mini")
    
    # Default job requirements
    if requirements is None:
        requirements = JobRequirements(
            title="Senior Python Developer",
            required_skills=["Python", "FastAPI", "PostgreSQL", "Git"],
            preferred_skills=["AWS", "Docker", "Kubernetes", "React"],
            min_experience_years=3,
            education_level="bachelor"
        )
    
    # Initialize workers
    workers = [
        PDFReaderWorker(),
        ResumeAnalyzer(llm=llm, requirements=requirements),
        CLIHumanProxyWorker(),  # For borderline case decisions
    ]
    
    orchestrator = Orchestrator(
        llm=llm,
        workers=workers,
        verbose=True,
        auto_save_path="./resume_screening_session.json"
    )
    
    return orchestrator, requirements


async def main():
    """Run the resume screener."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    # Get resume path from command line or use sample
    if len(sys.argv) > 1:
        resume_path = sys.argv[1]
        if not Path(resume_path).exists():
            print(f"Error: File not found: {resume_path}")
            return
    else:
        # Create a sample text file for demonstration
        resume_path = "./sample_resume.txt"
        Path(resume_path).write_text("""
JOHN DOE
Software Engineer | john.doe@email.com | (555) 123-4567

SUMMARY
Experienced Python developer with 4 years of experience building scalable web applications.
Strong background in FastAPI, Django, and cloud infrastructure.

SKILLS
- Languages: Python, JavaScript, SQL
- Frameworks: FastAPI, Django, React
- Databases: PostgreSQL, MongoDB, Redis
- Cloud: AWS (EC2, S3, Lambda), Docker
- Tools: Git, CI/CD, Terraform

EXPERIENCE

Senior Software Engineer | TechCorp Inc. | 2021-Present
- Designed and implemented microservices architecture using FastAPI
- Managed PostgreSQL databases with 10M+ records
- Deployed applications to AWS using Docker and Kubernetes

Software Engineer | StartupXYZ | 2019-2021
- Built RESTful APIs serving 100k daily requests
- Implemented automated testing with pytest
- Collaborated in agile team environment

EDUCATION
Bachelor of Science in Computer Science
State University, 2019
GPA: 3.7/4.0
""")
        print(f"Created sample resume at: {resume_path}")
    
    # Create orchestrator
    orchestrator, requirements = create_orchestrator(resume_path)
    
    # Define the screening goal
    goal = f"""
    Screen the resume at {resume_path} for the {requirements.title} position.
    
    Steps:
    1. Read the PDF/text file to extract resume content
    2. Analyze the resume against job requirements
    3. If the recommendation is "REVIEW", ask for human input
    4. Provide final hiring recommendation
    """
    
    # Create initial state with resume path
    state = Blackboard(goal=goal)
    state.metadata["resume_path"] = resume_path
    
    print("=" * 60)
    print("RESUME SCREENING AGENT")
    print("=" * 60)
    print(f"Resume: {resume_path}")
    print(f"Position: {requirements.title}")
    print("=" * 60)
    print()
    
    # Run the orchestrator
    result = await orchestrator.run(state=state, max_steps=8)
    
    # Display results
    print()
    print("=" * 60)
    print(f"STATUS: {result.status.value}")
    print(f"STEPS: {result.step_count}")
    print("=" * 60)
    
    # Display screening result
    for artifact in result.artifacts:
        if artifact.type == "screening_result":
            print("\n📋 SCREENING RESULT:\n")
            print(artifact.content)
    
    return result


if __name__ == "__main__":
    asyncio.run(main())
