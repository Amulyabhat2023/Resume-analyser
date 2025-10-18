from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class Resume(db.Model):
    __tablename__ = 'resumes'
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    raw_text = db.Column(db.Text)
    skills_json = db.Column(db.Text)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def skills(self):
        return json.loads(self.skills_json) if self.skills_json else []

    @skills.setter
    def skills(self, skills_list):
        self.skills_json = json.dumps(skills_list)

class Job(db.Model):
    __tablename__ = 'jobs'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(300), nullable=False)
    description = db.Column(db.Text)
    required_skills_json = db.Column(db.Text)

    @property
    def required_skills(self):
        return json.loads(self.required_skills_json) if self.required_skills_json else []

    @required_skills.setter
    def required_skills(self, skills_list):
        self.required_skills_json = json.dumps(skills_list)

class Analysis(db.Model):
    __tablename__ = 'analyses'
    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(db.Integer, db.ForeignKey('resumes.id'))
    job_id = db.Column(db.Integer, db.ForeignKey('jobs.id'))
    match_score = db.Column(db.Float)
    recommendations=db.Column(db.Text)
    skill_analysis=db.Column(db.Text)

    def to_dict(self):
        return {
            "id": self.id,
            "resume_id": self.resume_id,
            "job_id": self.job_id,
            "match_score": self.match_score,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
