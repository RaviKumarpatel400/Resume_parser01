import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_mysql_connector import MySQL

from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
import matplotlib.pyplot as plt
from resume_parser import parse_resume, get_job_description, calculate_similarity, find_non_matching_skills

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')  # Use environment variable for security

# Configure MySQL connection using environment variables
app.config['MYSQL_USER'] = os.getenv('MYSQL_USER', 'root')
app.config['MYSQL_PASSWORD'] = os.getenv('MYSQL_PASSWORD', 'Ramkr@123')
app.config['MYSQL_DATABASE'] = os.getenv('MYSQL_DATABASE', 'DAML')
app.config['MYSQL_HOST'] = os.getenv('MYSQL_HOST', 'localhost')

mysql = MySQL(app)
bcrypt = Bcrypt(app)

# Global lists to store details
resume_details = []
similarity_scores = []
pie_chart_images = []
non_matching_skills_list = []

# Switch backend for matplotlib to 'Agg'
plt.switch_backend('Agg')

# Add admin credentials (can be stored in the database for production)
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin123')  # Store admin password in an environment variable for security

@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['admin_loggedin'] = True
            flash('Admin login successful!', 'success')
            return redirect(url_for('dashboard'))  # Redirect to dashboard if credentials are correct
        else:
            flash('Invalid admin username or password', 'danger')

    return render_template('admin_login.html')

@app.route('/logout_admin')
def logout_admin():
    session.pop('admin_loggedin', None)
    flash('Admin logged out successfully', 'success')
    return redirect(url_for('admin_login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Hash the password for security
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

        # Insert the user into the database
        cursor = mysql.connection.cursor()
        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", (username, email, hashed_password))
        mysql.connection.commit()
        cursor.close()

        flash('Signup successful! You can now login.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Retrieve user from the database
        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        cursor.close()

        if user and bcrypt.check_password_hash(user[3], password):
            session['loggedin'] = True
            session['username'] = user[1]
            session['email'] = user[2]  # Store email in session
            session['user_id'] = user[0]  # Storing user ID in the session
            flash(f'Welcome {user[1]}!', 'success')
            return redirect(url_for('index'))  # Redirect to index after login
        else:
            flash('Login failed! Please check your email and password.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('username', None)
    session.pop('user_id', None)
    flash('You have successfully logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'loggedin' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    global resume_details, similarity_scores, pie_chart_images, non_matching_skills_list

    if request.method == 'POST':
        job_url = request.form['job_url']
        resumes = request.files.getlist('resumes')

        # Clear previous results
        resume_details.clear()
        similarity_scores.clear()
        pie_chart_images.clear()
        non_matching_skills_list.clear()

        job_description = get_job_description(job_url)

        for idx, resume in enumerate(resumes):
            details = parse_resume(resume)
            resume_details.append(details)

            # Calculate similarity for each resume
            skills_score, education_score, overall_score = calculate_similarity(details, job_description)
            similarity_scores.append({
                'skills_score': skills_score,
                'education_score': education_score,
                'overall_score': overall_score
            })

            # Find non-matching skills
            non_matching_skills = find_non_matching_skills(details['skills'], job_description)
            non_matching_skills_list.append(non_matching_skills)

            # Generate pie chart
            labels = ['Similarity', 'Difference']
            sizes = [overall_score, 100 - overall_score]
            colors = ['#D5AAFF', '#4ECDC4']

            plt.figure(figsize=(6, 6))
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')

            # Save the pie chart as an image in the static folder
            chart_filename = f'similarity_pie_chart_{idx}.png'
            chart_path = os.path.join('static', chart_filename)
            plt.savefig(chart_path)
            plt.close()

            # Store the path for rendering
            pie_chart_images.append(chart_filename)

        return redirect(url_for('candidate_details'))

    return render_template('index.html')

@app.route('/candidate_details')
def candidate_details():
    if not resume_details:
        return redirect(url_for('index'))

    return render_template('candidate_details.html', resume_details=resume_details, non_matching_skills=non_matching_skills_list)

@app.route('/dashboard')
def dashboard():
    if 'admin_loggedin' not in session:
        return redirect(url_for('admin_login'))  # Redirect to admin login if not logged in as admin

    # Retrieve all users from the database
    cursor = mysql.connection.cursor()
    cursor.execute("SELECT id, username, email FROM users")
    candidates = cursor.fetchall()
    cursor.close()

    # Pass candidates to the template
    return render_template('dashboard.html', candidates=candidates)

@app.route('/similarity_score')
def similarity_score():
    if not resume_details or not similarity_scores:
        return redirect(url_for('index'))

    return render_template('similarity_score.html', resume_details=resume_details, similarity_scores=similarity_scores)

@app.route('/visualization_graph')
def visualization_graph():
    if not pie_chart_images:
        return redirect(url_for('index'))

    return render_template('visualization_graph.html', pie_chart_images=pie_chart_images)

@app.route('/similarity_bar_chart')
def similarity_bar_chart():
    if not similarity_scores:
        return redirect(url_for('index'))

    candidates = [f'Candidate {i+1}' for i in range(len(similarity_scores))]
    skills_scores = [score['skills_score'] for score in similarity_scores]
    education_scores = [score['education_score'] for score in similarity_scores]
    overall_scores = [score['overall_score'] for score in similarity_scores]

    plt.figure(figsize=(10, 6))
    bar_width = 0.25
    index = range(len(candidates))

    plt.bar(index, skills_scores, width=bar_width, label='Skills', color='blue')
    plt.bar([i + bar_width for i in index], education_scores, width=bar_width, label='Education', color='green')
    plt.bar([i + 2 * bar_width for i in index], overall_scores, width=bar_width, label='Overall Similarity', color='purple')

    plt.xlabel('Candidates')
    plt.ylabel('Scores (%)')
    plt.title('Similarity Scores by Category')
    plt.xticks([i + bar_width for i in index], candidates)

    plt.legend()
    bar_chart_filename = 'similarity_bar_chart.png'
    bar_chart_path = os.path.join('static', bar_chart_filename)
    plt.savefig(bar_chart_path)
    plt.close()

    return render_template('similarity_bar_chart.html', bar_chart_image=bar_chart_filename)
