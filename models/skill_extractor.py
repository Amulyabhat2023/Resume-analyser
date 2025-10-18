"""
Skill Extraction Module

Uses NLP to extract skills, education, experience and contact info from resumes and jobs.
"""

import re
import os
import logging
from typing import List, Dict, Any
import PyPDF2
import docx
import spacy
import nltk
from nltk.tokenize import sent_tokenize
import nltk
import os

# Explicitly set the local nltk_data directory path (ensure it exists)
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.mkdir(nltk_data_path)

# Download punkt resource to this folder if not already present
nltk.download('punkt', download_dir=nltk_data_path)

# Tell NLTK to look for data in this local folder
nltk.data.path.append(nltk_data_path)

logger = logging.getLogger(__name__)

class SkillExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except Exception:
            self.nlp = None
        self.skill_keywords = self._load_skill_keywords()

    def _load_skill_keywords(self):
        return {
            'programming': {'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'swift', 'kotlin'},
            'frameworks': {'react', 'angular', 'vue', 'django', 'flask', 'spring', 'node.js', 'express'},
            'databases': {'mysql', 'postgresql', 'mongodb', 'sqlite', 'redis'},
            'cloud': {'aws', 'azure', 'gcp', 'heroku', 'digitalocean'},
            'devops': {'docker', 'kubernetes', 'jenkins', 'git', 'github'},
            'data_science': {'machine learning', 'deep learning', 'spark', 'tableau', 'power bi'},
            'soft_skills': {'leadership', 'communication', 'teamwork', 'problem solving'}
        }

    def extract_text_from_file(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return self._extract_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return self._extract_docx(file_path)
        elif ext == '.txt':
            return self._extract_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _extract_pdf(self, path):
        text = ""
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

    def _extract_docx(self, path):
        doc = docx.Document(path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text

    def _extract_txt(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def extract_skills(self, text: str) -> List[str]:
        found_skills = set()
        text_lower = text.lower()
        for category in self.skill_keywords:
            for skill in self.skill_keywords[category]:
                if skill in text_lower:
                    found_skills.add(skill.title())
        return sorted(found_skills)

    def extract_resume_info(self, text: str) -> Dict[str, Any]:
        info = {
            'skills': self.extract_skills(text),
            'education': self._extract_education(text),
            'experience': self._extract_experience(text),
            'email': self._extract_email(text),
            'phone': self._extract_phone(text),
            'name': self._extract_name(text),
        }
        return info

    def _extract_education(self, text):
        # Simple heuristic: lines containing education keywords
        ed_keywords = ['bachelor', 'master', 'phd', 'diploma', 'degree', 'university', 'college']
        lines = text.lower().split('\n')
        ed_info = [line for line in lines if any(kw in line for kw in ed_keywords)]
        return ed_info[:3]

    def _extract_experience(self, text):
        exp_keywords = ['experience', 'years', 'worked', 'employed']
        lines = text.lower().split('\n')
        exp_info = [line for line in lines if any(kw in line for kw in exp_keywords)]
        return exp_info[:3]

    def _extract_email(self, text):
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(pattern, text)
        return matches[0] if matches else ""

    def _extract_phone(self, text):
        pattern = r'\+?[0-9][0-9\s\-\(\)]{7,}[0-9]'
        matches = re.findall(pattern, text)
        return matches[0] if matches else ""

    def _extract_name(self, text):
        sentences = sent_tokenize(text)
        for sentence in sentences[:5]:
            words = sentence.split()
            if len(words) >= 2 and all(w[0].isupper() for w in words[:2]):
                return sentence
        return "Unknown"
    
    def extract_job_requirements(self, job_description: str) -> Dict[str, Any]:
        skills = self.extract_skills(job_description)
        return {'required_skills': skills}
