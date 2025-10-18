"""
Job Matcher Module

This module provides:
- Analysis of skill gaps between candidate and job requirements
- Recommendations for missing skills
- Scoring and matching logic for resumes and job descriptions
"""

import logging

logger = logging.getLogger(__name__)

class JobMatcher:
    def __init__(self):
        # Placeholder for any initialization
        pass

    def analyze_skill_gaps(self, candidate_skills, required_skills):
        """
        Analyze the skill gaps between candidate skills and job required skills.

        Args:
            candidate_skills (list[str]): Skills listed in candidate resume.
            required_skills (list[str]): Skills required by the job.

        Returns:
            dict: {
                'matched_skills': list of skills candidate has that match job,
                'missing_skills': list of required skills candidate lacks,
                'additional_skills': list of candidate skills not required by job,
                'total_required': total count of required skills,
                'match_percentage': float match percentage
            }
        """
        candidate_set = set(skill.lower() for skill in candidate_skills)
        required_set = set(skill.lower() for skill in required_skills)

        matched = candidate_set.intersection(required_set)
        missing = required_set.difference(candidate_set)
        additional = candidate_set.difference(required_set)

        total_required = len(required_set)
        match_percentage = (len(matched) / total_required * 100) if total_required > 0 else 0.0

        logger.info(f"Matched skills: {matched}")
        logger.info(f"Missing skills: {missing}")

        return {
            'matched_skills': [skill.title() for skill in matched],
            'missing_skills': [skill.title() for skill in missing],
            'additional_skills': [skill.title() for skill in additional],
            'total_required': total_required,
            'match_percentage': round(match_percentage, 2)
        }

    def generate_recommendations(self, skill_analysis, resume_category=None):
        """
        Generate personalized recommendations based on skill gaps.

        Args:
            skill_analysis (dict): Output from analyze_skill_gaps().
            resume_category (str, optional): Candidate's resume category for tailored suggestions.

        Returns:
            list: List of recommendation strings.
        """
        recommendations = []

        missing_skills = skill_analysis.get('missing_skills', [])
        match_pct = skill_analysis.get('match_percentage', 0)

        if match_pct >= 75:
            recommendations.append("Your skills strongly match the job requirements.")
            recommendations.append("Consider highlighting relevant projects and accomplishments.")
        elif 50 <= match_pct < 75:
            recommendations.append("Good skill match, but consider gaining expertise in some missing areas.")
        else:
            recommendations.append("Significant skill gaps detected; focused upskilling recommended.")

        # Suggest popular courses or certifications (example mappings)
        skill_to_courses = {
            'Node.js': ['Node.js Complete Guide', 'Express.js Fundamentals'],
            'AWS': ['AWS Certified Solutions Architect', 'AWS Fundamentals'],
            'Docker': ['Docker Mastery', 'Containerization Fundamentals'],
            'Kubernetes': ['Certified Kubernetes Administrator', 'Kubernetes Basics'],
            'Machine Learning': ['Machine Learning Specialization - Stanford', 'Deep Learning.ai'],
        }

        for skill in missing_skills:
            course_list = skill_to_courses.get(skill, [])
            if course_list:
                rec = f"Consider learning {skill} via courses such as: {', '.join(course_list)}."
                recommendations.append(rec)
            else:
                recommendations.append(f"Consider gaining experience or training in {skill}.")

        # Optional category specific recs
        if resume_category:
            if resume_category.lower() == 'data science' and 'machine learning' in missing_skills:
                recommendations.append("Explore advanced machine learning courses to improve your profile.")

        return recommendations

    def score_match(self, skill_analysis, experience_match=0, education_match=0):
        """
        Calculate a total match score combining skills, experience and education.

        Args:
            skill_analysis (dict): Output from analyze_skill_gaps().
            experience_match (float): Matching score from experience (0 to 1).
            education_match (float): Matching score from education (0 to 1).

        Returns:
            float: total weighted match score (0-1).
        """
        skill_weight = 0.6
        experience_weight = 0.3
        education_weight = 0.1

        skill_score = skill_analysis.get('match_percentage', 0) / 100

        total_score = (skill_weight * skill_score) + \
                      (experience_weight * experience_match) + \
                      (education_weight * education_match)

        return round(total_score, 3)
