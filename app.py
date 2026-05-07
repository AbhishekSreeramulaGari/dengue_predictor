import os
import secrets
import json
import warnings
from datetime import datetime
from functools import wraps

import joblib
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import inspect, text
from werkzeug.security import generate_password_hash, check_password_hash

from ward_analysis import WardAnalyzer
from complaint_analyzer import ComplaintSeverityAnalyzer
from ward_mapping import get_ward_name, get_ward_options, get_wards_list
from improved_predictor import ImprovedDenguePredictor
from ai_assistant import DengueAIAssistant

try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
except Exception:
    pass

# --------------------------------------
# App configuration
# --------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///dengue_users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['AUTH_ROLE_CODE'] = os.environ.get('AUTH_ROLE_CODE', 'AUTHORITY2026')

# --------------------------------------
# Database models
# --------------------------------------
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='public')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def is_authority(self):
        return self.role == 'authority'


class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ward_id = db.Column(db.String(50), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    reporter_name = db.Column(db.String(120), nullable=True)
    reporter_email = db.Column(db.String(120), nullable=False, index=True)
    severity_score = db.Column(db.Float, nullable=False, default=0.0)
    status = db.Column(db.String(32), nullable=False, default='open')
    assigned_to = db.Column(db.String(120), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


def _ensure_database_columns():
    inspector = inspect(db.engine)

    def ensure_column(table_name, column_name, alter_sql):
        if table_name not in inspector.get_table_names():
            return
        columns = [column['name'] for column in inspector.get_columns(table_name)]
        if column_name not in columns:
            db.session.execute(text(alter_sql))
            db.session.commit()

    ensure_column('user', 'role', "ALTER TABLE user ADD COLUMN role VARCHAR(50) DEFAULT 'public'")
    ensure_column('complaint', 'severity_score', "ALTER TABLE complaint ADD COLUMN severity_score FLOAT DEFAULT 0.0")
    ensure_column('complaint', 'status', "ALTER TABLE complaint ADD COLUMN status VARCHAR(32) DEFAULT 'open'")
    ensure_column('complaint', 'assigned_to', "ALTER TABLE complaint ADD COLUMN assigned_to VARCHAR(120)")
    ensure_column('complaint', 'notes', "ALTER TABLE complaint ADD COLUMN notes TEXT")
    ensure_column('complaint', 'created_at', "ALTER TABLE complaint ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP")
    ensure_column('complaint', 'updated_at', "ALTER TABLE complaint ADD COLUMN updated_at DATETIME DEFAULT CURRENT_TIMESTAMP")

with app.app_context():
    db.create_all()
    _ensure_database_columns()


# --------------------------------------
# Initialize components
# --------------------------------------
predictor = ImprovedDenguePredictor()
ai_assistant = DengueAIAssistant()
ward_analyzer = WardAnalyzer()
complaint_analyzer = ComplaintSeverityAnalyzer()


# --------------------------------------
# Helpers and authorization
# --------------------------------------

def get_current_user():
    email = session.get('user')
    if not email:
        return None
    return User.query.filter_by(email=email).first()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash('Please log in to continue.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def role_required(required_role):
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            user = get_current_user()
            if not user:
                flash('Please log in to continue.', 'warning')
                return redirect(url_for('login'))
            if user.role != required_role:
                flash('You do not have permission to access this page.', 'danger')
                return redirect(url_for('home'))
            return f(*args, **kwargs)
        return wrapped
    return decorator


def public_only(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        user = get_current_user()
        if user and user.is_authority():
            flash('This page is for public users only.', 'warning')
            return redirect(url_for('authority_dashboard'))
        return f(*args, **kwargs)
    return wrapped


# --------------------------------------
# Application routes
# --------------------------------------

@app.route('/')
def home():
    user = get_current_user()
    return render_template('index.html', user=user)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session['user'] = user.email
            session['role'] = user.role
            flash('Logged in successfully.', 'success')

            # Redirect based on role
            if user.is_authority():
                return redirect(url_for('authority_dashboard'))
            else:
                return redirect(url_for('public_dashboard'))

        flash('Invalid email or password.', 'danger')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        role_code = request.form.get('role_code', '').strip()

        if not email or not password:
            flash('Email and password are required.', 'danger')
            return render_template('register.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')

        if User.query.filter_by(email=email).first():
            flash('A user with that email already exists.', 'danger')
            return render_template('register.html')

        user_role = 'authority' if role_code == app.config['AUTH_ROLE_CODE'] else 'public'
        user = User(email=email, role=user_role)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        session['user'] = user.email
        session['role'] = user.role
        flash('Registration successful.', 'success')

        # Redirect based on role
        if user_role == 'authority':
            return redirect(url_for('authority_dashboard'))
        else:
            return redirect(url_for('public_dashboard'))

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))


# --------------------------------------
# Public User Routes
# --------------------------------------

@app.route('/dashboard')
@login_required
def public_dashboard():
    user = get_current_user()
    if user.is_authority():
        return redirect(url_for('authority_dashboard'))

    # Get user's complaints
    complaints = Complaint.query.filter_by(reporter_email=user.email)\
                               .order_by(Complaint.created_at.desc()).limit(5).all()

    # Get ward statistics
    ward_stats = get_ward_statistics()

    return render_template('public_dashboard.html',
                         user=user,
                         complaints=complaints,
                         ward_stats=ward_stats)


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    user = get_current_user()
    if user.is_authority():
        return redirect(url_for('authority_dashboard'))

    prediction_result = None
    ward_options = get_ward_options()

    if request.method == 'POST':
        ward_id = request.form.get('ward_id', '').strip()
        garbage = request.form.get('garbage', '0')
        waterlogging = request.form.get('waterlogging', '0')
        garbage_text = request.form.get('garbage_text', '').strip()
        waterlogging_text = request.form.get('waterlogging_text', '').strip()

        try:
            garbage_value = float(garbage)
            waterlogging_value = float(waterlogging)

            # Use improved predictor
            prediction_result = predictor.predict_comprehensive(
                garbage_value, waterlogging_value, garbage_text, waterlogging_text, int(ward_id) if ward_id else None
            )

            # Add ward name
            if ward_id:
                prediction_result['ward_name'] = get_ward_name(ward_id)

        except ValueError:
            flash('Please enter valid numeric values for complaint counts.', 'danger')

    return render_template('predict.html',
                         prediction=prediction_result,
                         ward_options=ward_options,
                         user=user)


@app.route('/report-complaint', methods=['GET', 'POST'])
@login_required
def report_complaint():
    user = get_current_user()
    if user.is_authority():
        return redirect(url_for('authority_dashboard'))

    success_message = None
    ward_options = get_ward_options()

    if request.method == 'POST':
        ward_id = request.form.get('ward_id', '').strip()
        category = request.form.get('category', 'Other').strip()
        description = request.form.get('description', '').strip()

        if not ward_id or not description:
            flash('Ward and description are required.', 'danger')
            return render_template('report_complaint.html', ward_options=ward_options, user=user)

        severity_score, _ = complaint_analyzer.analyze_complaint_text(description)
        complaint = Complaint(
            ward_id=ward_id,
            category=category,
            description=description,
            reporter_name=user.email,
            reporter_email=user.email,
            severity_score=severity_score,
            status='open'
        )

        db.session.add(complaint)
        db.session.commit()

        success_message = f'Your complaint for {get_ward_name(ward_id)} has been registered successfully. Our authority team will review it shortly.'

    return render_template('report_complaint.html',
                         success=success_message,
                         ward_options=ward_options,
                         user=user)


@app.route('/my-complaints')
@login_required
def my_complaints():
    user = get_current_user()
    if user.is_authority():
        return redirect(url_for('authority_dashboard'))

    complaints = Complaint.query.filter_by(reporter_email=user.email)\
                               .order_by(Complaint.created_at.desc()).all()

    # Add ward names to complaints
    for complaint in complaints:
        complaint.ward_name = get_ward_name(complaint.ward_id)

    return render_template('my_complaints.html', complaints=complaints, user=user)


@app.route('/ai-assistant', methods=['GET', 'POST'])
@login_required
def public_ai_assistant():
    user = get_current_user()
    if user.is_authority():
        return redirect(url_for('authority_ai_assistant'))

    assistance = None
    ward_options = get_ward_options()

    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        ward_id = request.form.get('ward_id', '')

        context = {}
        if ward_id:
            context['ward_name'] = get_ward_name(ward_id)

        assistance = ai_assistant.get_public_assistance(query, context)

    return render_template('public_ai_assistant.html',
                         assistance=assistance,
                         ward_options=ward_options,
                         user=user)


# --------------------------------------
# Authority/Admin Routes
# --------------------------------------

@app.route('/authority/dashboard')
@role_required('authority')
def authority_dashboard():
    user = get_current_user()

    # Get complaint statistics
    complaints = Complaint.query.order_by(Complaint.created_at.desc()).all()
    summary = {
        'open': Complaint.query.filter_by(status='open').count(),
        'in_progress': Complaint.query.filter_by(status='in_progress').count(),
        'fixed': Complaint.query.filter_by(status='fixed').count(),
        'closed': Complaint.query.filter_by(status='closed').count(),
        'total': len(complaints)
    }

    # Get recent complaints with ward names
    recent_complaints = complaints[:10]
    for complaint in recent_complaints:
        complaint.ward_name = get_ward_name(complaint.ward_id)

    # Get ward-wise complaint density
    ward_complaints = {}
    for complaint in complaints:
        ward_name = get_ward_name(complaint.ward_id)
        if ward_name not in ward_complaints:
            ward_complaints[ward_name] = 0
        ward_complaints[ward_name] += 1

    top_wards = sorted(ward_complaints.items(), key=lambda x: x[1], reverse=True)[:5]

    return render_template('authority_dashboard.html',
                         complaints=recent_complaints,
                         summary=summary,
                         top_wards=top_wards,
                         user=user)


@app.route('/authority/complaints', methods=['GET', 'POST'])
@role_required('authority')
def authority_complaints():
    user = get_current_user()

    # Handle status updates
    if request.method == 'POST':
        complaint_id = request.form.get('complaint_id')
        status = request.form.get('status', '').strip()
        notes = request.form.get('notes', '').strip()

        complaint = Complaint.query.get(complaint_id)
        if complaint and status in ['open', 'in_progress', 'fixed', 'closed']:
            complaint.status = status
            complaint.notes = notes
            complaint.assigned_to = user.email
            db.session.commit()
            flash('Complaint status updated.', 'success')
        else:
            flash('Unable to update complaint. Please try again.', 'danger')

    # Get filter parameters
    ward_filter = request.args.get('ward', '')
    status_filter = request.args.get('status', '')
    category_filter = request.args.get('category', '')

    # Build query
    query = Complaint.query

    if ward_filter:
        query = query.filter_by(ward_id=ward_filter)
    if status_filter:
        query = query.filter_by(status=status_filter)
    if category_filter:
        query = query.filter_by(category=category_filter)

    complaints = query.order_by(Complaint.created_at.desc()).all()

    # Add ward names
    for complaint in complaints:
        complaint.ward_name = get_ward_name(complaint.ward_id)

    # Get filter options
    ward_options = get_ward_options()
    status_options = [('open', 'Open'), ('in_progress', 'In Progress'),
                     ('fixed', 'Fixed'), ('closed', 'Closed')]
    category_options = db.session.query(Complaint.category.distinct()).all()
    category_options = [(cat[0], cat[0]) for cat in category_options]

    return render_template('authority_complaints.html',
                         complaints=complaints,
                         ward_options=ward_options,
                         status_options=status_options,
                         category_options=category_options,
                         filters={'ward': ward_filter, 'status': status_filter, 'category': category_filter},
                         user=user)


@app.route('/authority/analytics')
@role_required('authority')
def authority_analytics():
    user = get_current_user()

    # Get ward statistics
    ward_stats = get_ward_statistics()

    # Get complaint trends
    complaints_by_month = db.session.query(
        db.func.strftime('%Y-%m', Complaint.created_at).label('month'),
        db.func.count(Complaint.id).label('count')
    ).group_by('month').order_by('month').all()

    # Get status distribution
    status_counts = db.session.query(
        Complaint.status,
        db.func.count(Complaint.id).label('count')
    ).group_by(Complaint.status).all()

    return render_template('authority_analytics.html',
                         ward_stats=ward_stats,
                         complaints_by_month=complaints_by_month,
                         status_counts=status_counts,
                         user=user)


@app.route('/authority/ai-assistant', methods=['GET', 'POST'])
@role_required('authority')
def authority_ai_assistant():
    user = get_current_user()

    assistance = None

    if request.method == 'POST':
        query = request.form.get('query', '').strip()

        # Get context from current system state
        context = get_system_context()

        assistance = ai_assistant.get_admin_assistance(query, context)

    return render_template('authority_ai_assistant.html',
                         assistance=assistance,
                         user=user)


# --------------------------------------
# API Routes
# --------------------------------------

@app.route('/api/ward-stats')
def api_ward_stats():
    """API endpoint for ward statistics"""
    stats = get_ward_statistics()
    return jsonify(stats)


@app.route('/api/complaint-stats')
@role_required('authority')
def api_complaint_stats():
    """API endpoint for complaint statistics"""
    stats = {
        'total': Complaint.query.count(),
        'open': Complaint.query.filter_by(status='open').count(),
        'in_progress': Complaint.query.filter_by(status='in_progress').count(),
        'fixed': Complaint.query.filter_by(status='fixed').count(),
        'closed': Complaint.query.filter_by(status='closed').count()
    }
    return jsonify(stats)


# --------------------------------------
# Helper Functions
# --------------------------------------

def get_ward_statistics():
    """Get ward-level statistics"""
    try:
        ward_df = pd.read_csv('dengue_ward_summary.csv')
        if not ward_df.empty:
            ward_df = ward_df.rename(columns={
                'Ward_ID': 'ward_id',
                'Total_Cases': 'total_cases',
                'Avg_Cases': 'avg_cases'
            })

            # Add ward names
            ward_df['ward_name'] = ward_df['ward_id'].apply(get_ward_name)

            ward_summary = ward_df.sort_values('avg_cases', ascending=False).to_dict('records')
            return ward_summary
    except Exception:
        pass

    return []


def get_system_context():
    """Get current system context for AI assistant"""
    complaint_count = Complaint.query.count()
    open_complaints = Complaint.query.filter_by(status='open').count()

    # Get high-risk wards (wards with most complaints)
    ward_complaints = db.session.query(
        Complaint.ward_id,
        db.func.count(Complaint.id).label('count')
    ).group_by(Complaint.ward_id).order_by(db.desc('count')).limit(5).all()

    high_risk_wards = [get_ward_name(wc[0]) for wc in ward_complaints]

    return {
        'complaint_count': complaint_count,
        'open_complaints': open_complaints,
        'high_risk_wards': high_risk_wards
    }


# --------------------------------------
# Error handling
# --------------------------------------
@app.errorhandler(404)
def page_not_found(error):
    return render_template('error.html', message='Page not found.'), 404


@app.errorhandler(403)
def forbidden(error):
    return render_template('error.html', message='Forbidden access.'), 403


if __name__ == '__main__':
    app.run(debug=True)
