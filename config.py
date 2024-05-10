import os
from enum import Enum
from diskcache import UNKNOWN
from dotenv import load_dotenv
from dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Dict, Optional, List

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


MODELS = {
    "LLama3": "mario-llama:latest",
    "Llama2": "llama2",
    "Gemma": "gemma",
    "Mistral": "mistral",
}


# -------- Entity Extraction -----------
class Entity(str, Enum):
    LANGUAGE = "LANGUAGE"
    DOMAIN = "DOMAIN"
    TOOL = "TOOL"
    FRAMEWORK = "FRAMEWORK"
    DATABASE = "DATABASE"
    PROFESSION = "PROFESSION"
    ORGANIZATION = "ORGANIZATION"
    UNKNOWN = "UNKNOWN"


class NER(BaseModel):
    """Extracted Entity and its type from the document."""

    entity: str = Field(description="the detected entity in the document")
    type: Entity = Field(description="the type of the detected entity")

    class Config:
        json_schema_extra = {"additionalProperties": False}


class Entities(BaseModel):
    response: List[NER] = Field(
        description="List of extracted entities like language, domain, tool, framework, database, and profession from the document. If the entity type is not one of the mentioned type, it will be marked as UNKNOWN."
    )

    class Config:
        json_schema_extra = {"additionalProperties": False}


@dataclass
class NERPrompt:
    SYSTEM_PROMPT: str = "You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{schema}\n</schema>"
    USER_PROMPT: str = "I am applying to this job, here is the job description: {job_description} \n\n From job description, please extract the entities like language, domain, tool, framework, database, and profession. If the entity type is not one of the mentioned type, it will be marked as UNKNOWN. Return the response as JSON list containing the extracted entities."


# -------- Resume Generation -----------
@dataclass
class GenerationPrompt:
    SYSTEM_PROMPT: str = "You are Snapjobs AI, a Resume Generator Bot. Given the job description and the user's resume bullet points your task is to generate tailored resume bullet points that align with the job description. Answer as Resume Generation Bot, the assistant only."
    USER_PROMPT: str = "Here is the job description: \n\n{job_description} \n\n Draft bullet points: {resume_bullet_point}. \n\n Generate tailored resume bullet points that align with the job description. Give me only four resume bullet points which I can add into my Resume. Give me only and only four resume bullet points which I can add into my Resume."


# -------- Job Description Parsing -----------
class JobDescription(BaseModel):
    """Extracted Job Description."""

    employment_type: Optional[str] = Field(description="Employment type of the job")
    seniority: Optional[str] = Field(description="Seniority level of the job")
    job_category: Optional[str] = Field(description="Category of the job")
    location: Optional[str] = Field(description="Location of the job")
    salary: Optional[str] = Field(description="Salary of the job")
    min_experience: Optional[str] = Field(
        description="Minimum experience required for the job"
    )
    skills_required: List[str] = Field(description="Skills required for the job")
    job_requirements: List[str] = Field(description="Other requirements for the job")

    class Config:
        json_schema_extra = {"additionalProperties": False}


@dataclass
class JobDescriptionPrompt:
    SYSTEM_PROMPT: str = NERPrompt.SYSTEM_PROMPT
    USER_PROMPT: str = "I am applyig to this job, here is the job description: {job_description} \n\n Extract job details like employment type, seniority, job category, location, salary, minimum experience, skills required and job requirements. Return the response as JSON containing the extracted job details."
