import fitz  # PyMuPDF
import re
import nltk
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup

# Ensure NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))

# Function to extract email
def extract_email(text):
    email = re.findall(r'[\w\.-]+@[\w\.-]+', text)
    return email[0] if email else 'N/A'

# Function to extract phone number
def extract_phone(text):
    phone = re.findall(r'\+?\d[\d -]{8,}\d', text)
    return phone[0] if phone else 'N/A'

# Function to extract name
def extract_name(text):
    lines = text.strip().split('\n')
    for line in lines[:10]:
        if re.match(r'^[A-Z][a-zA-Z]+\s[A-Z][a-zA-Z]+', line):
            return line.strip()
    for line in lines[:10]:
        if re.search(r'(Name|Full Name):\s*(.*)', line, re.IGNORECASE):
            return line.split(':')[-1].strip()
    return 'N/A'

# Function to extract education
def extract_education(text):
    education = []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if "university" in line.lower() or "college" in line.lower() or "degree" in line.lower():
            education.append(line.strip())
        elif "bachelor" in line.lower() or "master" in line.lower() or "ph.d" in line.lower():
            education.append(line.strip())
            if i + 1 < len(lines):
                education.append(lines[i + 1].strip())
    return ', '.join(education) if education else 'N/A'

# Function to extract skills
def extract_skills(text):
    skills_section = re.search(r'(Skills|Technical Skills|Key Skills)([\s\S]*?)(Experience|Projects|Education|Certifications|$)', text, re.IGNORECASE)
    if skills_section:
        skills_text = skills_section.group(2)
        skills = re.split(r'[\n,;]', skills_text)
        skills = [skill.strip() for skill in skills if len(skill.strip()) > 1 and skill.strip().lower() not in STOP_WORDS]
        return set(skills)
    return set()

# Function to extract experience
def extract_experience(text):
    experience_section = re.search(r'(Experience|Work History|Employment|Professional Experience)([\s\S]*?)(Education|Skills|Certifications|Projects|$)', text, re.IGNORECASE)
    if experience_section:
        experience_text = experience_section.group(2)
        experience = re.split(r'[\n]', experience_text)
        experience = [exp.strip() for exp in experience if len(exp.strip()) > 1]
        return ', '.join(experience)
    return 'N/A'

# Function to extract projects
def extract_projects(text):
    projects_section = re.findall(r'(Projects|Project Work|Key Projects)([\s\S]*?)(Experience|Education|Skills|Certifications)', text, re.IGNORECASE)
    if projects_section:
        projects = re.split(r'\n', projects_section[0][1])
        return ', '.join(project.strip() for project in projects if len(project.strip()) > 0)
    return 'N/A'

# Main resume parsing function
def parse_resume(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ''
    for page in doc:
        text += page.get_text()

    details = {
        'name': extract_name(text),
        'email': extract_email(text),
        'phone': extract_phone(text),
        'skills': extract_skills(text),
        'education': extract_education(text),
        'projects': extract_projects(text),
        'experience': extract_experience(text)  # Add experience extraction
    }
    return details

# Function to preprocess text
def preprocess(text):
    words = re.sub(r'\W+', ' ', text).lower().split()
    return ' '.join([word for word in words if word not in STOP_WORDS])

# Function to calculate similarity
def calculate_similarity(resume_details, job_description):
    resume_text = preprocess(resume_details['education'])
    skills_text = ', '.join(resume_details['skills'])
    resume_skills_text = preprocess(skills_text)
    job_skills_text = preprocess(job_description)
    
    vectorizer = TfidfVectorizer()

    # Education Similarity
    tfidf_matrix_education = vectorizer.fit_transform([resume_text, job_description])
    education_similarity = cosine_similarity(tfidf_matrix_education[0:1], tfidf_matrix_education[1:2])[0][0]

    # Skills Similarity
    tfidf_matrix_skills = vectorizer.fit_transform([resume_skills_text, job_skills_text])
    skills_similarity = cosine_similarity(tfidf_matrix_skills[0:1], tfidf_matrix_skills[1:2])[0][0]

    overall_similarity = (skills_similarity + education_similarity) / 2

    return round(skills_similarity * 100, 2), round(education_similarity * 100, 2), round(overall_similarity * 100, 2)

# Function to get job description from URL
def get_job_description(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    job_desc = soup.find(class_="jobDescriptionText")
    if job_desc:
        return job_desc.get_text()

    return soup.get_text()

# Function to find non-matching skills
def find_non_matching_skills(resume_skills, job_description):
    # Extract skills from the job description
    job_skills = extract_skills(job_description)

    # Find non-matching skills
    non_matching_skills = job_skills.difference(resume_skills)
    return non_matching_skills
