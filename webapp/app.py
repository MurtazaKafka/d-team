"""
D-Team Formation Web Application
================================

Flask-based web interface for the D-Team formation solver.
Provides file upload, solver execution, results visualization, and export.

Author: DCI Team Formation System
Date: January 2026
"""

# =============================================================================
# Imports
# =============================================================================

import os
import sys
import io
import uuid
import math
import traceback
from datetime import datetime

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

# Add parent directory to path to import solver
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dteam_solver_v2 import DTeamSolverV2


# =============================================================================
# Flask App Configuration
# =============================================================================

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# In-memory cache for solutions (in production, use Redis or database)
solutions_cache = {}


# =============================================================================
# Utility Functions
# =============================================================================

def clean_for_json(obj):
    """
    Recursively clean an object for JSON serialization.
    Handles NaN, Infinity, numpy types, and pandas NA values.
    """
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.integer, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif pd.isna(obj):
        return None
    return obj


def safe_get(d, key, default=None):
    """
    Safely get a value from a dict.
    Returns default if value is None, NaN, or missing.
    """
    val = d.get(key, default)
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass  # pd.isna can fail on some types like lists
    return val


# =============================================================================
# Routes
# =============================================================================

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload and run the solver.
    
    Expects:
        - file: Excel file with registrant data
        - min_team_size: Minimum team size (default: 8)
        - max_team_size: Maximum team size (default: 10)
        - allow_flexible: Allow flexible team sizes (default: true)
        - time_limit: Solver time limit in seconds (default: 300)
    
    Returns:
        JSON with formatted solution or error message
    """
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith(('.xlsx', '.xls')):
            return jsonify({'error': 'Please upload an Excel file (.xlsx or .xls)'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Get solver parameters
        min_team_size = int(request.form.get('min_team_size', 8))
        max_team_size = int(request.form.get('max_team_size', 10))
        allow_flexible = request.form.get('allow_flexible', 'true') == 'true'
        time_limit = int(request.form.get('time_limit', 300))
        
        # Run solver
        solver = DTeamSolverV2(filepath, verbose=False)
        solution = solver.solve(
            min_team_size=min_team_size,
            max_team_size=max_team_size,
            allow_flexible_size=allow_flexible,
            time_limit_seconds=time_limit
        )
        
        # Generate solution ID and format for frontend
        solution_id = str(uuid.uuid4())
        formatted = format_solution_for_frontend(solution)
        formatted['solution_id'] = solution_id
        formatted['filename'] = filename
        formatted['timestamp'] = datetime.now().isoformat()
        
        # Cache for later download
        solutions_cache[solution_id] = {
            'solution': solution,
            'solver': solver,
            'filepath': filepath
        }
        
        return jsonify(clean_for_json(formatted))
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/download/<solution_id>/<format_type>')
def download_solution(solution_id, format_type):
    """
    Download the solution in the requested format.
    
    Args:
        solution_id: UUID of the cached solution
        format_type: 'excel' or 'csv'
    
    Returns:
        File download response
    """
    if solution_id not in solutions_cache:
        return jsonify({'error': 'Solution not found'}), 404
    
    cached = solutions_cache[solution_id]
    solution = cached['solution']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format_type == 'excel':
        return generate_excel_download(solution, timestamp)
    elif format_type == 'csv':
        return generate_csv_download(solution, timestamp)
    else:
        return jsonify({'error': 'Invalid format'}), 400


# =============================================================================
# Solution Formatting
# =============================================================================

def format_solution_for_frontend(solution):
    """
    Transform solver output into frontend-friendly format.
    Includes all participant data for comprehensive display.
    """
    stats = solution['statistics']
    
    teams = []
    for tid, team in sorted(solution['teams'].items()):
        c = team['composition']
        
        # Check constraint violations
        hard_violations, soft_violations = check_team_violations(team, c)
        
        status = 'success' if not hard_violations and not soft_violations else \
                 'error' if hard_violations else 'warning'
        
        teams.append({
            'id': tid,
            'day': team['info']['day'],
            'time': team['info']['time'],
            'format': 'Virtual' if team['info']['is_virtual'] else 'In-Person',
            'size': team['size'],
            'primary_facilitator': team['primary_facilitator']['id'] if team['primary_facilitator'] else None,
            'secondary_facilitator': team['secondary_facilitator']['id'] if team['secondary_facilitator'] else None,
            'composition': c,
            'status': status,
            'hard_violations': hard_violations,
            'soft_violations': soft_violations,
            'members': [format_member(m) for m in team['members']]
        })
    
    # Format unassigned participants
    unassigned = [
        {
            'id': p['id'],
            'is_student': p['is_student'],
            'format_pref': p['format_pref'],
            'total_available': p['total_available'],
            'available_slots': p['available_slots'],
        }
        for p in solution['unassigned_participants']
    ]
    
    # Format friend pairs
    friend_pairs_satisfied = [
        {'p1': p1, 'p2': p2, 'team': tid}
        for p1, p2, tid in solution['friend_pairs_satisfied']
    ]
    friend_pairs_unsatisfied = [
        {'p1': p1, 'p2': p2}
        for p1, p2 in solution['friend_pairs_unsatisfied']
    ]
    
    return {
        'status': solution['status'],
        'statistics': {
            'total_participants': stats['total_participants'],
            'assigned': stats['assigned'],
            'unassigned': stats['unassigned'],
            'assignment_rate': round(stats['assignment_rate'], 1),
            'teams_formed': stats['teams_formed'],
            'friend_pairs_total': stats['friend_pairs_total'],
            'friend_pairs_satisfied': stats['friend_pairs_satisfied'],
        },
        'teams': teams,
        'unassigned_participants': unassigned,
        'friend_pairs_satisfied': friend_pairs_satisfied,
        'friend_pairs_unsatisfied': friend_pairs_unsatisfied,
        'constraint_violations': solution['constraint_violations'],
    }


def check_team_violations(team, composition):
    """Check a team for hard and soft constraint violations."""
    hard_violations = []
    soft_violations = []
    
    # Hard constraints
    if composition['students'] < 2:
        hard_violations.append(f"Only {composition['students']} students (need ≥2)")
    if composition['non_students'] < 2:
        hard_violations.append(f"Only {composition['non_students']} non-students (need ≥2)")
    if team['primary_facilitator'] is None:
        hard_violations.append("Missing primary facilitator")
    if team['secondary_facilitator'] is None:
        hard_violations.append("Missing secondary facilitator")
    
    # Soft constraints
    if team['size'] < 8:
        soft_violations.append(f"Size {team['size']} < 8")
    if team['size'] > 10:
        soft_violations.append(f"Size {team['size']} > 10")
    if composition['women'] < 2:
        soft_violations.append(f"Only {composition['women']} women (want ≥2)")
    if composition['men'] < 2:
        soft_violations.append(f"Only {composition['men']} men (want ≥2)")
    if composition['conservatives'] < 1:
        soft_violations.append("No conservatives")
    if composition['liberals'] < 1:
        soft_violations.append("No liberals")
    if composition['white'] < 1:
        soft_violations.append("No white participants")
    if composition['non_white'] < 1:
        soft_violations.append("No non-white participants")
    
    return hard_violations, soft_violations


def format_member(m):
    """Format a single team member with all available data."""
    return {
        'id': m['id'],
        
        # Binary flags for solver constraints
        'is_student': m['is_student'],
        'is_female': m['is_female'],
        'is_male': m['is_male'],
        'is_conservative': m['is_conservative'],
        'is_liberal': m['is_liberal'],
        'is_white': m['is_white'],
        'is_nonwhite': m['is_nonwhite'],
        
        # Detailed category
        'category': safe_get(m, 'category', 'Unknown'),
        'is_staff': safe_get(m, 'is_staff', False),
        'is_faculty': safe_get(m, 'is_faculty', False),
        'is_alum': safe_get(m, 'is_alum', False),
        'is_community_alum': safe_get(m, 'is_community_alum', False),
        
        # Detailed gender
        'is_nonbinary': safe_get(m, 'is_nonbinary', False),
        'is_trans': safe_get(m, 'is_trans', False),
        'prefer_not_say_gender': safe_get(m, 'prefer_not_say_gender', False),
        
        # Detailed race
        'races': safe_get(m, 'races', []),
        'race_black': safe_get(m, 'race_black', False),
        'race_hispanic': safe_get(m, 'race_hispanic', False),
        'race_white': safe_get(m, 'race_white', False),
        'race_asian': safe_get(m, 'race_asian', False),
        'race_native': safe_get(m, 'race_native', False),
        'race_other': safe_get(m, 'race_other', False),
        'prefer_not_say_race': safe_get(m, 'prefer_not_say_race', False),
        
        # Ideology
        'ideology': safe_get(m, 'ideology', ''),
        'is_moderate': safe_get(m, 'is_moderate', False),
        
        # Issue positions
        'issue1_position': safe_get(m, 'issue1_position', ''),
        'issue2_position': safe_get(m, 'issue2_position', ''),
        'issue1_agree': safe_get(m, 'issue1_agree', False),
        'issue1_disagree': safe_get(m, 'issue1_disagree', False),
        'issue2_agree': safe_get(m, 'issue2_agree', False),
        'issue2_disagree': safe_get(m, 'issue2_disagree', False),
        
        # Demographics
        'year': safe_get(m, 'year', ''),
        'age_range': safe_get(m, 'age_range', ''),
        
        # Format preference
        'format_pref': safe_get(m, 'format_pref', ''),
        'format_preference_full': safe_get(m, 'format_preference_full', ''),
        
        # Course credit
        'taking_for_credit': safe_get(m, 'taking_for_credit', False),
        'courses': safe_get(m, 'courses', []),
        
        # Connection/referral
        'connection': safe_get(m, 'connection', ''),
        
        # Registration info
        'date_submitted': safe_get(m, 'date_submitted', ''),
        'status': safe_get(m, 'status', ''),
        
        # Friend info
        'friend_invited': safe_get(m, 'friend_invited', None),
        'friend_invited_by': safe_get(m, 'friend_invited_by', None),
        'invited_friend_importance': safe_get(m, 'invited_friend_importance', ''),
        'been_invited_importance': safe_get(m, 'been_invited_importance', ''),
        
        # Availability
        'total_available': safe_get(m, 'total_available', 0),
    }


# =============================================================================
# Export Functions
# =============================================================================

def generate_excel_download(solution, timestamp):
    """
    Generate comprehensive Excel file with multiple detailed sheets:
    - Overview: High-level statistics
    - Team Summary: Team composition breakdown
    - All Participants: Complete participant profiles with team assignments
    - Friend Pairs: Friend request outcomes
    - Unassigned: Participants who couldn't be placed
    """
    output = io.BytesIO()
    
    # =========================================================================
    # Sheet 1: Overview Statistics
    # =========================================================================
    stats = solution['statistics']
    overview_data = {
        'Metric': [
            'Total Registered Participants',
            'Participants Assigned to Teams',
            'Participants Unassigned',
            'Assignment Rate',
            'Total Teams Formed',
            'Friend Pair Requests',
            'Friend Pairs Satisfied',
            'Friend Pair Success Rate',
        ],
        'Value': [
            stats['total_participants'],
            stats['assigned'],
            stats['unassigned'],
            f"{stats['assignment_rate']:.1f}%",
            stats['teams_formed'],
            stats['friend_pairs_total'],
            stats['friend_pairs_satisfied'],
            f"{(stats['friend_pairs_satisfied'] / stats['friend_pairs_total'] * 100) if stats['friend_pairs_total'] > 0 else 0:.1f}%",
        ]
    }
    df_overview = pd.DataFrame(overview_data)
    
    # =========================================================================
    # Sheet 2: Team Summary
    # =========================================================================
    summary_rows = []
    for tid, team in sorted(solution['teams'].items()):
        c = team['composition']
        summary_rows.append({
            'Team ID': tid,
            'Day': team['info']['day'],
            'Time': team['info']['time'],
            'Format': 'Virtual' if team['info']['is_virtual'] else 'In-Person',
            'Team Size': team['size'],
            'Primary Facilitator ID': team['primary_facilitator']['id'] if team['primary_facilitator'] else '',
            'Secondary Facilitator ID': team['secondary_facilitator']['id'] if team['secondary_facilitator'] else '',
            '# Students': c['students'],
            '# Non-Students': c['non_students'],
            '# Women': c['women'],
            '# Men': c['men'],
            '# Conservative': c['conservatives'],
            '# Liberal': c['liberals'],
            '# Moderate': c['moderates'],
            '# White': c['white'],
            '# Non-White': c['non_white'],
            '# Issue1 Agree': c.get('issue1_agree', 0),
            '# Issue1 Disagree': c.get('issue1_disagree', 0),
            '# Issue2 Agree': c.get('issue2_agree', 0),
            '# Issue2 Disagree': c.get('issue2_disagree', 0),
        })
    df_summary = pd.DataFrame(summary_rows)
    
    # =========================================================================
    # Sheet 3: All Participants (Complete Profiles)
    # =========================================================================
    participant_rows = []
    for tid, team in sorted(solution['teams'].items()):
        for m in team['members']:
            # Determine role
            role = ''
            if team['primary_facilitator'] and m['id'] == team['primary_facilitator']['id']:
                role = 'Primary Facilitator'
            elif team['secondary_facilitator'] and m['id'] == team['secondary_facilitator']['id']:
                role = 'Secondary Facilitator'
            
            # Build gender string
            gender = 'Unknown'
            if m['is_female']:
                gender = 'Female'
            elif m['is_male']:
                gender = 'Male'
            elif m.get('is_nonbinary'):
                gender = 'Nonbinary'
            elif m.get('is_trans'):
                gender = 'Trans'
            elif m.get('prefer_not_say_gender'):
                gender = 'Prefer not to say'
            
            # Build race string
            races = m.get('races', [])
            race_str = ', '.join(races) if races else ('White' if m['is_white'] else ('Non-White' if m['is_nonwhite'] else 'Unknown'))
            
            # Build courses string
            courses = m.get('courses', [])
            courses_str = ', '.join(courses) if courses else ''
            
            participant_rows.append({
                # === TEAM ASSIGNMENT ===
                'Team ID': tid,
                'Team Day': team['info']['day'],
                'Team Time': team['info']['time'],
                'Team Format': 'Virtual' if team['info']['is_virtual'] else 'In-Person',
                'Facilitator Role': role,
                
                # === IDENTIFICATION ===
                'Participant ID': m['id'],
                'Registration Status': m.get('status', ''),
                'Date Submitted': m.get('date_submitted', ''),
                
                # === CATEGORY/TYPE ===
                'Category': m.get('category', 'Student' if m['is_student'] else 'Non-Student'),
                'Is Student': 'Yes' if m['is_student'] else 'No',
                'Is Faculty': 'Yes' if m.get('is_faculty') else 'No',
                'Is Staff': 'Yes' if m.get('is_staff') else 'No',
                'Is Alum': 'Yes' if m.get('is_alum') else 'No',
                'Is Community Alum': 'Yes' if m.get('is_community_alum') else 'No',
                
                # === DEMOGRAPHICS ===
                'Year/Class': m.get('year', ''),
                'Age Range': m.get('age_range', ''),
                'Gender': gender,
                'Race/Ethnicity': race_str,
                
                # === RACE BREAKDOWN ===
                'Is Black/African American': 'Yes' if m.get('race_black') else 'No',
                'Is Hispanic/Latino': 'Yes' if m.get('race_hispanic') else 'No',
                'Is White': 'Yes' if m.get('race_white') or m['is_white'] else 'No',
                'Is Asian': 'Yes' if m.get('race_asian') else 'No',
                'Is Native American': 'Yes' if m.get('race_native') else 'No',
                'Is Other Race': 'Yes' if m.get('race_other') else 'No',
                
                # === POLITICAL VIEWS ===
                'Political Ideology': m.get('ideology', ''),
                'Is Conservative': 'Yes' if m['is_conservative'] else 'No',
                'Is Liberal': 'Yes' if m['is_liberal'] else 'No',
                'Is Moderate': 'Yes' if m.get('is_moderate') else 'No',
                
                # === ISSUE POSITIONS ===
                'Immigration Issue Position': m.get('issue1_position', ''),
                'Immigration Issue Agree': 'Yes' if m.get('issue1_agree') else 'No',
                'Immigration Issue Disagree': 'Yes' if m.get('issue1_disagree') else 'No',
                'Presidency Issue Position': m.get('issue2_position', ''),
                'Presidency Issue Agree': 'Yes' if m.get('issue2_agree') else 'No',
                'Presidency Issue Disagree': 'Yes' if m.get('issue2_disagree') else 'No',
                
                # === FORMAT PREFERENCES ===
                'Format Preference Code': m.get('format_pref', ''),
                'Format Preference': 'In-Person Only' if m.get('format_pref') == 'P' else ('Virtual Only' if m.get('format_pref') == 'Z' else 'Either'),
                'Format Preference (Full)': m.get('format_preference_full', ''),
                
                # === AVAILABILITY ===
                'Total Available Time Slots': m.get('total_available', 0),
                
                # === COURSE CREDIT ===
                'Taking for Class Credit': 'Yes' if m.get('taking_for_credit') else 'No',
                'Courses': courses_str,
                
                # === REFERRAL/CONNECTION ===
                'How Heard About DCI': m.get('connection', ''),
                
                # === FRIEND REQUESTS ===
                'Friend Invited (ID)': m.get('friend_invited', ''),
                'Invited Friend Importance': m.get('invited_friend_importance', ''),
                'Invited By Friend (ID)': m.get('friend_invited_by', ''),
                'Being Invited Importance': m.get('been_invited_importance', ''),
            })
    df_participants = pd.DataFrame(participant_rows)
    
    # =========================================================================
    # Sheet 4: Friend Pairs
    # =========================================================================
    friend_rows = []
    for p1, p2, tid in solution['friend_pairs_satisfied']:
        friend_rows.append({
            'Participant 1 ID': p1,
            'Participant 2 ID': p2,
            'Status': 'Satisfied ✓',
            'Assigned Team': tid,
        })
    for p1, p2 in solution['friend_pairs_unsatisfied']:
        friend_rows.append({
            'Participant 1 ID': p1,
            'Participant 2 ID': p2,
            'Status': 'Not Satisfied ✗',
            'Assigned Team': 'N/A - Different teams or unassigned',
        })
    df_friends = pd.DataFrame(friend_rows) if friend_rows else pd.DataFrame(columns=['Participant 1 ID', 'Participant 2 ID', 'Status', 'Assigned Team'])
    
    # =========================================================================
    # Sheet 5: Unassigned Participants
    # =========================================================================
    unassigned_rows = []
    for p in solution['unassigned_participants']:
        unassigned_rows.append({
            'Participant ID': p['id'],
            'Category': 'Student' if p['is_student'] else 'Non-Student',
            'Format Preference': 'In-Person Only' if p['format_pref'] == 'P' else ('Virtual Only' if p['format_pref'] == 'Z' else 'Either'),
            'Total Available Slots': p['total_available'],
            'Available Time Slots': ', '.join(p['available_slots']),
            'Reason': 'Limited availability' if p['total_available'] <= 5 else 'No compatible team found',
        })
    df_unassigned = pd.DataFrame(unassigned_rows) if unassigned_rows else pd.DataFrame(columns=['Participant ID', 'Category', 'Format Preference', 'Total Available Slots', 'Available Time Slots', 'Reason'])
    
    # =========================================================================
    # Write all sheets to Excel
    # =========================================================================
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_overview.to_excel(writer, sheet_name='Overview', index=False)
        df_summary.to_excel(writer, sheet_name='Team Summary', index=False)
        df_participants.to_excel(writer, sheet_name='All Participants', index=False)
        df_friends.to_excel(writer, sheet_name='Friend Pairs', index=False)
        df_unassigned.to_excel(writer, sheet_name='Unassigned', index=False)
        
        # Auto-adjust column widths for better readability
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Cap at 50
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'dteam_assignments_{timestamp}.xlsx'
    )


def generate_csv_download(solution, timestamp):
    """
    Generate comprehensive CSV file with all participant data.
    Single flat file with complete information for each assigned participant.
    """
    output = io.StringIO()
    
    rows = []
    for tid, team in sorted(solution['teams'].items()):
        for m in team['members']:
            # Determine role
            role = ''
            if team['primary_facilitator'] and m['id'] == team['primary_facilitator']['id']:
                role = 'Primary Facilitator'
            elif team['secondary_facilitator'] and m['id'] == team['secondary_facilitator']['id']:
                role = 'Secondary Facilitator'
            
            # Build gender string
            gender = 'Unknown'
            if m['is_female']:
                gender = 'Female'
            elif m['is_male']:
                gender = 'Male'
            elif m.get('is_nonbinary'):
                gender = 'Nonbinary'
            
            # Build race string
            races = m.get('races', [])
            race_str = ', '.join(races) if races else ('White' if m['is_white'] else ('Non-White' if m['is_nonwhite'] else 'Unknown'))
            
            # Build courses string
            courses = m.get('courses', [])
            courses_str = ', '.join(courses) if courses else ''
            
            rows.append({
                # Team info
                'Team ID': tid,
                'Team Day': team['info']['day'],
                'Team Time': team['info']['time'],
                'Team Format': 'Virtual' if team['info']['is_virtual'] else 'In-Person',
                'Facilitator Role': role,
                
                # Participant info
                'Participant ID': m['id'],
                'Category': m.get('category', 'Student' if m['is_student'] else 'Non-Student'),
                'Year/Class': m.get('year', ''),
                'Age Range': m.get('age_range', ''),
                'Gender': gender,
                'Race/Ethnicity': race_str,
                
                # Political views
                'Political Ideology': m.get('ideology', ''),
                'Is Conservative': 'Yes' if m['is_conservative'] else 'No',
                'Is Liberal': 'Yes' if m['is_liberal'] else 'No',
                'Is Moderate': 'Yes' if m.get('is_moderate') else 'No',
                
                # Issue positions
                'Immigration Issue': m.get('issue1_position', ''),
                'Presidency Issue': m.get('issue2_position', ''),
                
                # Preferences
                'Format Preference': 'In-Person Only' if m.get('format_pref') == 'P' else ('Virtual Only' if m.get('format_pref') == 'Z' else 'Either'),
                'Available Slots': m.get('total_available', 0),
                
                # Course credit
                'For Class Credit': 'Yes' if m.get('taking_for_credit') else 'No',
                'Courses': courses_str,
                
                # Connection
                'How Heard About DCI': m.get('connection', ''),
                
                # Friend info
                'Friend Invited': m.get('friend_invited', ''),
                'Invited By': m.get('friend_invited_by', ''),
                
                # Registration
                'Status': m.get('status', ''),
                'Date Submitted': m.get('date_submitted', ''),
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'dteam_assignments_{timestamp}.csv'
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
