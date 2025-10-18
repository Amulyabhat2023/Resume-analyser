from flask import Flask, render_template, request, jsonify
from models import db, Resume, Job, Analysis
import os
import json
from datetime import datetime

from config import Config

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

# Load your ML models appropriately (implement these modules)
from models.resume_classifier import ResumeClassifier
from models.skill_extractor import SkillExtractor
from models.similarity_engine import SimilarityEngine
from models.job_matcher import JobMatcher

resume_classifier = ResumeClassifier()
resume_classifier.load_model()

skill_extractor = SkillExtractor()
similarity_engine = SimilarityEngine()
job_matcher = JobMatcher()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files or request.files['resume'].filename == '':
        return jsonify({'error': 'No resume file selected'}), 400
    
    file = request.files['resume']
    job_desc = request.form.get('job_description', '')
    
    filename = file.filename
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(upload_path)

    resume_text = skill_extractor.extract_text_from_file(upload_path)
    cleaned_resume = resume_text
    cleaned_job_desc = job_desc

    resume_info = skill_extractor.extract_resume_info(cleaned_resume)
    job_reqs = skill_extractor.extract_job_requirements(cleaned_job_desc)
    resume_category = resume_classifier.predict_category(cleaned_resume)['category']
    match_score = similarity_engine.calculate_similarity(cleaned_resume, cleaned_job_desc)
    skill_analysis = job_matcher.analyze_skill_gaps(resume_info['skills'], job_reqs['required_skills'])
    recommendations = job_matcher.generate_recommendations(skill_analysis, resume_category)

    resume_record = Resume(filename=filename, raw_text=resume_text, skills_json=json.dumps(resume_info['skills']))
    db.session.add(resume_record)
    db.session.commit()

    job_record = Job(title="Job Title", description=job_desc, required_skills_json=json.dumps(job_reqs['required_skills']))
    db.session.add(job_record)
    db.session.commit()

    analysis_record = Analysis(resume_id=resume_record.id, job_id=job_record.id, match_score=match_score,recommendations=json.dumps(recommendations),skill_analysis=json.dumps(skill_analysis))
    db.session.add(analysis_record)
    db.session.commit()

    return jsonify({
        'match_score': match_score,
        'resume_category': resume_category,
        'skill_analysis': skill_analysis,
        'recommendations': recommendations,
        'analysis_id': analysis_record.id
    })

from flask import render_template

@app.route('/results/<int:analysis_id>')
def show_results(analysis_id):
    analysis_record = Analysis.query.get_or_404(analysis_id)
    recommendations = json.loads(analysis_record.recommendations) if analysis_record.recommendations else []
    skill_analysis = json.loads(analysis_record.skill_analysis) if analysis_record.skill_analysis else {}

    return render_template(
        'results.html',
        match_score=analysis_record.match_score,
        recommendations=recommendations,
        skill_analysis=skill_analysis,
        resume_id=analysis_record.resume_id,
        job_id=analysis_record.job_id,
        created_at=analysis_record.created_at
    )


if __name__ == '__main__':
    app.run(debug=Config.DEBUG)
