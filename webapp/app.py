"""
D-Team Formation Web Application
================================

Flask-based web interface for the D-Team formation solver.
Provides file upload, solver execution, results visualization, and export.
Supports both CSV and Excel data files.

Author: DCI Team Formation System
Date: February 2026
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
from dteam_solver_v3 import DTeamSolverV3


# =============================================================================
# Flask App Configuration
# =============================================================================

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# In-memory cache for solutions
solutions_cache = {}


# =============================================================================
# Utility Functions
# =============================================================================

def clean_for_json(obj):
    """Recursively clean an object for JSON serialization."""
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
    """Safely get a value from a dict, handling NaN/None."""
    val = d.get(key, default)
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    try:
        if pd.isna(val):
            return default
    except (TypeError, ValueError):
        pass
    return val


def detect_data_format(filepath):
    """
    Detect whether the file is in Qualtrics format (v3) or old format (v2).
    Supports both CSV and Excel files.
    """
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        columns = set(df.columns)

        # Qualtrics format indicators
        qualtrics_indicators = ['Q5', 'Q8', 'Q10', 'Rounds', 'Pro Liberty', 'Pro Rule of Law', 'Format']
        qualtrics_avail = ['Fri_1', 'Fri1_1', 'Fri2_1', 'Mon_1', 'Sat_1']

        # Old format indicators
        old_indicators = ['Unique ID', 'student', 'male', 'female', 'immp', 'presp', 'portrait']
        old_avail = ['m1030', 't1230', 'w1030', 'f1030']

        qualtrics_score = sum(1 for col in qualtrics_indicators if col in columns) + \
                          sum(1 for col in qualtrics_avail if col in columns)
        old_score = sum(1 for col in old_indicators if col in columns) + \
                    sum(1 for col in old_avail if col in columns)

        if qualtrics_score > old_score:
            return 'v3'
        elif old_score > 0:
            return 'v2'
        else:
            return 'v3'

    except Exception as e:
        print(f"Error detecting format: {e}")
        return 'v2'


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
    Supports both CSV and Excel files.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not file.filename.endswith(('.xlsx', '.xls', '.csv')):
            return jsonify({'error': 'Please upload an Excel (.xlsx/.xls) or CSV (.csv) file'}), 400

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

        # Detect data format and use appropriate solver
        data_format = detect_data_format(filepath)
        print(f"Detected data format: {data_format}")

        if data_format == 'v3':
            solver = DTeamSolverV3(filepath, verbose=False)
        else:
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
    """Download the solution in the requested format."""
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
    Includes all participant data, fellow data, team analytics, and facilitator analytics.
    """
    stats = solution['statistics']

    teams = []
    for tid, team in sorted(solution['teams'].items()):
        c = team['composition']
        hard_violations, soft_violations = check_team_violations(team, c)

        status = 'success' if not hard_violations and not soft_violations else \
                 'error' if hard_violations else 'warning'

        round_type = team['info'].get('round_type', 'B')
        round_name = {'B': 'Both Rounds', 'C': 'First Round Only', 'D': 'Second Round Only'}.get(round_type, 'Both Rounds')

        # Format primary/secondary facilitator data
        pf = team['primary_facilitator']
        sf = team['secondary_facilitator']

        teams.append({
            'id': tid,
            'day': team['info']['day'],
            'time': team['info']['time'],
            'format': 'Virtual' if team['info']['is_virtual'] else 'In-Person',
            'round_type': round_type,
            'round_name': round_name,
            'size': team['size'],
            'primary_facilitator': format_facilitator_for_team(pf) if pf else None,
            'secondary_facilitator': format_facilitator_for_team(sf) if sf else None,
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
            'category': safe_get(p, 'category', ''),
            'format_pref': p['format_pref'],
            'format_preference_full': safe_get(p, 'format_preference_full', ''),
            'total_available': p['total_available'],
            'total_available_if_necessary': p.get('total_available_if_necessary', 0),
            'available_slots': p['available_slots'],
            'round_type': p.get('round_type', 'B'),
            'round_name': {'B': 'Both', 'C': '1st Only', 'D': '2nd Only'}.get(p.get('round_type', 'B'), 'Both'),
            'ideology': safe_get(p, 'ideology', ''),
            'year': safe_get(p, 'year', ''),
            'age_range': safe_get(p, 'age_range', ''),
            'source': safe_get(p, 'source', ''),
        }
        for p in solution['unassigned_participants']
    ]

    # Friend pairs
    friend_pairs_satisfied = [
        {'p1': p1, 'p2': p2, 'team': tid}
        for p1, p2, tid in solution['friend_pairs_satisfied']
    ]
    friend_pairs_unsatisfied = [
        {'p1': p1, 'p2': p2}
        for p1, p2 in solution['friend_pairs_unsatisfied']
    ]

    # Build statistics
    statistics = {
        'total_participants': stats['total_participants'],
        'assigned': stats['assigned'],
        'unassigned': stats['unassigned'],
        'assignment_rate': round(stats['assignment_rate'], 1),
        'teams_formed': stats['teams_formed'],
        'friend_pairs_total': stats['friend_pairs_total'],
        'friend_pairs_satisfied': stats['friend_pairs_satisfied'],
    }

    # Add v3-specific stats
    if 'assigned_to_preferred_slot' in stats:
        statistics['assigned_to_preferred_slot'] = stats['assigned_to_preferred_slot']
        statistics['assigned_to_if_necessary_slot'] = stats['assigned_to_if_necessary_slot']

    if 'total_fellows' in stats:
        statistics['total_fellows'] = stats['total_fellows']
        statistics['fellows_assigned'] = stats['fellows_assigned']
        statistics['new_fellows_assigned'] = stats.get('new_fellows_assigned', 0)
        statistics['returning_fellows_assigned'] = stats.get('returning_fellows_assigned', 0)

    # Teams by round
    round_counts = {'B': 0, 'C': 0, 'D': 0}
    for team in teams:
        rt = team.get('round_type', 'B')
        round_counts[rt] = round_counts.get(rt, 0) + 1
    statistics['teams_by_round'] = {
        'both': round_counts.get('B', 0),
        'first_only': round_counts.get('C', 0),
        'second_only': round_counts.get('D', 0),
    }

    # ========== TEAM ANALYTICS ==========
    team_analytics = build_team_analytics(teams)

    # ========== FACILITATOR ANALYTICS ==========
    facilitator_analytics = build_facilitator_analytics(solution)

    # ========== FELLOWS DATA (for table view) ==========
    fellows_data = []
    if 'fellows_data' in solution:
        fellows_data = solution['fellows_data']

    return {
        'status': solution['status'],
        'statistics': statistics,
        'teams': teams,
        'unassigned_participants': unassigned,
        'friend_pairs_satisfied': friend_pairs_satisfied,
        'friend_pairs_unsatisfied': friend_pairs_unsatisfied,
        'constraint_violations': solution['constraint_violations'],
        'team_analytics': team_analytics,
        'facilitator_analytics': facilitator_analytics,
        'fellows_data': fellows_data,
    }


def format_facilitator_for_team(f):
    """Format a facilitator record for team display."""
    if f is None:
        return None
    return {
        'id': f['id'],
        'is_fellow': True,
        'facilitator_role': safe_get(f, 'facilitator_role', ''),
        'was_facilitator_before': safe_get(f, 'was_facilitator_before', False),
        'was_online_last_semester': safe_get(f, 'was_online_last_semester', False),
        'was_not_primary_last_semester': safe_get(f, 'was_not_primary_last_semester', False),
        'category': safe_get(f, 'category', 'Fellow'),
        'year': safe_get(f, 'year', ''),
        'age_range': safe_get(f, 'age_range', ''),
        'ideology': safe_get(f, 'ideology', ''),
        'gender_code': safe_get(f, 'gender_code', 0),
        'is_male': safe_get(f, 'is_male', False),
        'is_female': safe_get(f, 'is_female', False),
        'is_student': safe_get(f, 'is_student', False),
        'races': safe_get(f, 'races', []),
        'issue1_position': safe_get(f, 'issue1_position', ''),
        'issue2_position': safe_get(f, 'issue2_position', ''),
        'source': safe_get(f, 'source', ''),
        'format_preference_full': safe_get(f, 'format_preference_full', ''),
        'rounds_full': safe_get(f, 'rounds_full', ''),
    }


def build_team_analytics(teams):
    """Build aggregate team analytics data."""
    if not teams:
        return {}

    sizes = [t['size'] for t in teams]
    analytics = {
        'avg_team_size': round(sum(sizes) / len(sizes), 1) if sizes else 0,
        'min_team_size': min(sizes) if sizes else 0,
        'max_team_size': max(sizes) if sizes else 0,
        'total_teams': len(teams),
        'teams_with_hard_violations': sum(1 for t in teams if t['hard_violations']),
        'teams_with_soft_violations': sum(1 for t in teams if t['soft_violations']),
        'teams_ok': sum(1 for t in teams if not t['hard_violations'] and not t['soft_violations']),
        'virtual_teams': sum(1 for t in teams if t['format'] == 'Virtual'),
        'inperson_teams': sum(1 for t in teams if t['format'] == 'In-Person'),
        'teams_by_day': {},
        'composition_averages': {},
        'size_distribution': {},
    }

    # Teams by day
    day_counts = {}
    for t in teams:
        day = t['day']
        day_counts[day] = day_counts.get(day, 0) + 1
    analytics['teams_by_day'] = day_counts

    # Size distribution
    for s in sizes:
        key = str(s)
        analytics['size_distribution'][key] = analytics['size_distribution'].get(key, 0) + 1

    # Composition averages
    comp_keys = ['students', 'non_students', 'women', 'men', 'conservatives', 'liberals',
                 'moderates', 'white', 'non_white', 'issue1_agree', 'issue1_disagree',
                 'issue2_agree', 'issue2_disagree']
    for key in comp_keys:
        values = [t['composition'].get(key, 0) for t in teams]
        analytics['composition_averages'][key] = round(sum(values) / len(values), 1) if values else 0

    return analytics


def build_facilitator_analytics(solution):
    """Build facilitator/fellow analytics data."""
    fa_list = solution.get('facilitator_assignments', [])
    if not fa_list:
        return {}

    total = len(fa_list)
    assigned = [f for f in fa_list if f['total_assignments'] > 0]
    unassigned = [f for f in fa_list if f['total_assignments'] == 0]
    new_fellows = [f for f in fa_list if not f['was_facilitator_before']]
    returning = [f for f in fa_list if f['was_facilitator_before']]
    new_assigned = [f for f in assigned if not f['was_facilitator_before']]
    returning_assigned = [f for f in assigned if f['was_facilitator_before']]

    analytics = {
        'total_fellows': total,
        'fellows_assigned': len(assigned),
        'fellows_unassigned': len(unassigned),
        'new_fellows_total': len(new_fellows),
        'new_fellows_assigned': len(new_assigned),
        'returning_fellows_total': len(returning),
        'returning_fellows_assigned': len(returning_assigned),
        'assignment_rate': round(len(assigned) / total * 100, 1) if total > 0 else 0,
        'fellows_detail': [],
    }

    # Detailed fellow list
    for fa in fa_list:
        f = fa['fellow']
        analytics['fellows_detail'].append({
            'id': f['id'],
            'category': safe_get(f, 'category', ''),
            'was_facilitator_before': fa['was_facilitator_before'],
            'was_online_last_semester': fa.get('was_online_last_semester', False),
            'was_not_primary_last_semester': fa.get('was_not_primary_last_semester', False),
            'primary_teams': fa['primary_teams'],
            'secondary_teams': fa['secondary_teams'],
            'total_assignments': fa['total_assignments'],
            'ideology': safe_get(f, 'ideology', ''),
            'gender_code': safe_get(f, 'gender_code', 0),
            'year': safe_get(f, 'year', ''),
        })

    return analytics


def check_team_violations(team, composition):
    """Check a team for hard and soft constraint violations."""
    hard_violations = []
    soft_violations = []

    if composition['students'] < 2:
        hard_violations.append(f"Only {composition['students']} students (need ≥2)")
    if composition['non_students'] < 2:
        hard_violations.append(f"Only {composition['non_students']} non-students (need ≥2)")
    if team['primary_facilitator'] is None:
        hard_violations.append("Missing primary facilitator")
    if team['secondary_facilitator'] is None:
        hard_violations.append("Missing secondary facilitator")

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
    """Format a single team member with ALL available data."""
    gender_map = {1: 'Male', 2: 'Female', 3: 'Non-binary', 4: 'Prefer not to say', 5: 'Self-describe'}
    gender_code = safe_get(m, 'gender_code', 0)

    return {
        'id': m['id'],

        # Binary flags
        'is_student': m['is_student'],
        'is_female': m['is_female'],
        'is_male': m['is_male'],
        'is_conservative': m['is_conservative'],
        'is_liberal': m['is_liberal'],
        'is_white': m['is_white'],
        'is_nonwhite': m['is_nonwhite'],

        # Facilitator info
        'is_fellow': safe_get(m, 'is_fellow', False),
        'facilitator_role': safe_get(m, 'facilitator_role', None),

        # Category
        'category': safe_get(m, 'category', 'Unknown'),
        'is_staff': safe_get(m, 'is_staff', False),
        'is_faculty': safe_get(m, 'is_faculty', False),
        'is_alum': safe_get(m, 'is_alum', False),
        'is_community_alum': safe_get(m, 'is_community_alum', False),

        # Gender
        'gender': gender_map.get(gender_code, ''),
        'gender_code': gender_code,
        'is_nonbinary': safe_get(m, 'is_nonbinary', False),
        'is_trans': safe_get(m, 'is_trans', False),
        'prefer_not_say_gender': safe_get(m, 'prefer_not_say_gender', False),

        # Race
        'races': safe_get(m, 'races', []),
        'race_codes': safe_get(m, 'race_codes', []),
        'race_binary': safe_get(m, 'race_binary', 2),
        'race_black': safe_get(m, 'race_black', False),
        'race_hispanic': safe_get(m, 'race_hispanic', False),
        'race_white': safe_get(m, 'race_white', False),
        'race_asian': safe_get(m, 'race_asian', False),
        'race_native': safe_get(m, 'race_native', False),
        'race_other': safe_get(m, 'race_other', False),
        'prefer_not_say_race': safe_get(m, 'prefer_not_say_race', False),

        # Ideology
        'ideology': safe_get(m, 'ideology', ''),
        'ideology_code': safe_get(m, 'ideology_code', 0),
        'is_moderate': safe_get(m, 'is_moderate', False),

        # Issue positions
        'issue1_position': safe_get(m, 'issue1_position', ''),
        'issue2_position': safe_get(m, 'issue2_position', ''),
        'issue1_agree': safe_get(m, 'issue1_agree', False),
        'issue1_disagree': safe_get(m, 'issue1_disagree', False),
        'issue2_agree': safe_get(m, 'issue2_agree', False),
        'issue2_disagree': safe_get(m, 'issue2_disagree', False),
        'pro_liberty_code': safe_get(m, 'pro_liberty_code', 0),
        'pro_rule_code': safe_get(m, 'pro_rule_code', 0),

        # Demographics
        'year': safe_get(m, 'year', ''),
        'year_code': safe_get(m, 'year_code', 0),
        'age_range': safe_get(m, 'age_range', ''),
        'age_code': safe_get(m, 'age_code', 0),
        'source': safe_get(m, 'source', ''),
        'source_codes': safe_get(m, 'source_codes', []),

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
        'friend_invited_name': safe_get(m, 'friend_invited_name', ''),
        'friend_invited_by_name': safe_get(m, 'friend_invited_by_name', ''),
        'invited_friend_importance': safe_get(m, 'invited_friend_importance', ''),
        'been_invited_importance': safe_get(m, 'been_invited_importance', ''),

        # Availability
        'total_available': safe_get(m, 'total_available', 0),
        'total_available_if_necessary': safe_get(m, 'total_available_if_necessary', 0),
        'assigned_to_preferred': safe_get(m, 'assigned_to_preferred', True),

        # Round info
        'round_type': safe_get(m, 'round_type', 'B'),
        'rounds_full': safe_get(m, 'rounds_full', 'Both Rounds'),
    }


# =============================================================================
# Export Functions
# =============================================================================

def generate_excel_download(solution, timestamp):
    """Generate comprehensive Excel file with multiple sheets."""
    output = io.BytesIO()

    stats = solution['statistics']

    # ---- Sheet 1: Overview ----
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
            'Total Fellows',
            'Fellows Assigned',
            'New Fellows Assigned',
            'Returning Fellows Assigned',
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
            stats.get('total_fellows', ''),
            stats.get('fellows_assigned', ''),
            stats.get('new_fellows_assigned', ''),
            stats.get('returning_fellows_assigned', ''),
        ]
    }
    df_overview = pd.DataFrame(overview_data)

    # ---- Sheet 2: Team Summary ----
    summary_rows = []
    for tid, team in sorted(solution['teams'].items()):
        c = team['composition']
        round_type = team['info'].get('round_type', 'B')
        round_name = {'B': 'Both Rounds', 'C': 'First Round Only', 'D': 'Second Round Only'}.get(round_type, 'Both Rounds')
        pf = team['primary_facilitator']
        sf = team['secondary_facilitator']

        summary_rows.append({
            'Team ID': tid,
            'Day': team['info']['day'],
            'Time': team['info']['time'],
            'Format': 'Virtual' if team['info']['is_virtual'] else 'In-Person',
            'Round': round_name,
            'Team Size': team['size'],
            'Primary Facilitator ID': pf['id'] if pf else '',
            'Primary Facilitator Status': 'New' if pf and not pf.get('was_facilitator_before') else 'Returning' if pf else '',
            'Secondary Facilitator ID': sf['id'] if sf else '',
            'Secondary Facilitator Status': 'New' if sf and not sf.get('was_facilitator_before') else 'Returning' if sf else '',
            '# Students': c['students'],
            '# Non-Students': c['non_students'],
            '# Women': c['women'],
            '# Men': c['men'],
            '# Conservative': c['conservatives'],
            '# Liberal': c['liberals'],
            '# Moderate': c['moderates'],
            '# White': c['white'],
            '# Non-White': c['non_white'],
            '# Pro Liberty Agree': c.get('issue1_agree', 0),
            '# Pro Liberty Disagree': c.get('issue1_disagree', 0),
            '# Pro Rule of Law Agree': c.get('issue2_agree', 0),
            '# Pro Rule of Law Disagree': c.get('issue2_disagree', 0),
        })
    df_summary = pd.DataFrame(summary_rows)

    # ---- Sheet 3: All Participants ----
    gender_map = {1: 'Male', 2: 'Female', 3: 'Non-binary', 4: 'Prefer not to say', 5: 'Self-describe'}
    participant_rows = []
    for tid, team in sorted(solution['teams'].items()):
        for m in team['members']:
            role = ''
            if team['primary_facilitator'] and m['id'] == team['primary_facilitator']['id']:
                role = 'Primary'
            elif team['secondary_facilitator'] and m['id'] == team['secondary_facilitator']['id']:
                role = 'Secondary'

            gc = m.get('gender_code', 0)
            gender = gender_map.get(gc, 'Unknown') if gc else 'Unknown'
            if m['is_female']: gender = 'Female'
            elif m['is_male']: gender = 'Male'
            elif m.get('is_nonbinary'): gender = 'Non-binary'

            races = m.get('races', [])
            race_str = ', '.join(races) if races else ('White' if m['is_white'] else ('Non-White' if m['is_nonwhite'] else 'Unknown'))

            round_type = team['info'].get('round_type', 'B')
            round_name = {'B': 'Both Rounds', 'C': 'First Round Only', 'D': 'Second Round Only'}.get(round_type, 'Both Rounds')

            participant_rows.append({
                'Team ID': tid,
                'Team Day': team['info']['day'],
                'Team Time': team['info']['time'],
                'Team Format': 'Virtual' if team['info']['is_virtual'] else 'In-Person',
                'Team Round': round_name,
                'Is Facilitator': 'Yes' if role else 'No',
                'Facilitator Role': role,
                'Participant ID': m['id'],
                'Category': m.get('category', 'Student' if m['is_student'] else 'Non-Student'),
                'Is Student': 'Yes' if m['is_student'] else 'No',
                'Year/Class': m.get('year', ''),
                'Age Range': m.get('age_range', ''),
                'Gender': gender,
                'Race/Ethnicity': race_str,
                'Political Ideology': m.get('ideology', ''),
                'Is Conservative': 'Yes' if m['is_conservative'] else 'No',
                'Is Liberal': 'Yes' if m['is_liberal'] else 'No',
                'Is Moderate': 'Yes' if m.get('is_moderate') else 'No',
                'Pro Liberty Position': m.get('issue1_position', ''),
                'Pro Rule of Law Position': m.get('issue2_position', ''),
                'Format Preference': m.get('format_preference_full', ''),
                'Rounds Preference': m.get('rounds_full', ''),
                'Total Available Slots': m.get('total_available', 0),
                'How Heard About DCI': m.get('source', m.get('connection', '')),
                'Friend Invited (ID)': m.get('friend_invited', ''),
                'Invited By Friend (ID)': m.get('friend_invited_by', ''),
            })
    df_participants = pd.DataFrame(participant_rows)

    # ---- Sheet 4: Facilitators ----
    fac_rows = []
    for fa in solution.get('facilitator_assignments', []):
        f = fa['fellow']
        gc = f.get('gender_code', 0)
        gender = gender_map.get(gc, '') if gc else ''
        fac_rows.append({
            'Fellow ID': f['id'],
            'Category': f.get('category', ''),
            'Gender': gender,
            'Ideology': f.get('ideology', ''),
            'Was Facilitator Before': 'Yes' if fa['was_facilitator_before'] else 'No',
            'Was Online Last Semester': 'Yes' if fa.get('was_online_last_semester') else 'No',
            'Was Not Primary Last Semester': 'Yes' if fa.get('was_not_primary_last_semester') else 'No',
            'Primary Teams': ', '.join(str(t) for t in fa['primary_teams']),
            'Secondary Teams': ', '.join(str(t) for t in fa['secondary_teams']),
            'Total Assignments': fa['total_assignments'],
        })
    df_facilitators = pd.DataFrame(fac_rows) if fac_rows else pd.DataFrame()

    # ---- Sheet 5: Friend Pairs ----
    friend_rows = []
    for p1, p2, tid in solution['friend_pairs_satisfied']:
        friend_rows.append({'Participant 1': p1, 'Participant 2': p2, 'Status': 'Satisfied ✓', 'Team': tid})
    for p1, p2 in solution['friend_pairs_unsatisfied']:
        friend_rows.append({'Participant 1': p1, 'Participant 2': p2, 'Status': 'Not Satisfied ✗', 'Team': 'N/A'})
    df_friends = pd.DataFrame(friend_rows) if friend_rows else pd.DataFrame(columns=['Participant 1', 'Participant 2', 'Status', 'Team'])

    # ---- Sheet 6: Unassigned ----
    unassigned_rows = []
    for p in solution['unassigned_participants']:
        round_type = p.get('round_type', 'B')
        round_name = {'B': 'Both Rounds', 'C': 'First Round Only', 'D': 'Second Round Only'}.get(round_type, 'Both Rounds')
        total_avail = p['total_available'] + p.get('total_available_if_necessary', 0)
        if total_avail <= 3:
            reason = 'Very limited availability'
        elif round_type in ['C', 'D']:
            reason = f'Not enough participants for {round_name} teams'
        else:
            reason = 'No compatible team found'

        unassigned_rows.append({
            'Participant ID': p['id'],
            'Category': p.get('category', 'Student' if p['is_student'] else 'Non-Student'),
            'Format Preference': p.get('format_preference_full', ''),
            'Round Preference': round_name,
            'Available Slots': p['total_available'],
            'If-Necessary Slots': p.get('total_available_if_necessary', 0),
            'Reason': reason,
        })
    df_unassigned = pd.DataFrame(unassigned_rows) if unassigned_rows else pd.DataFrame()

    # Write all sheets
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_overview.to_excel(writer, sheet_name='Overview', index=False)
        df_summary.to_excel(writer, sheet_name='Team Summary', index=False)
        df_participants.to_excel(writer, sheet_name='All Participants', index=False)
        if not df_facilitators.empty:
            df_facilitators.to_excel(writer, sheet_name='Facilitators', index=False)
        df_friends.to_excel(writer, sheet_name='Friend Pairs', index=False)
        if not df_unassigned.empty:
            df_unassigned.to_excel(writer, sheet_name='Unassigned', index=False)

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
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

    output.seek(0)
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=f'dteam_assignments_{timestamp}.xlsx'
    )


def generate_csv_download(solution, timestamp):
    """Generate comprehensive CSV file."""
    output = io.StringIO()
    gender_map = {1: 'Male', 2: 'Female', 3: 'Non-binary', 4: 'Prefer not to say', 5: 'Self-describe'}

    rows = []
    for tid, team in sorted(solution['teams'].items()):
        for m in team['members']:
            role = ''
            if team['primary_facilitator'] and m['id'] == team['primary_facilitator']['id']:
                role = 'Primary'
            elif team['secondary_facilitator'] and m['id'] == team['secondary_facilitator']['id']:
                role = 'Secondary'

            gc = m.get('gender_code', 0)
            gender = gender_map.get(gc, 'Unknown') if gc else 'Unknown'
            if m['is_female']: gender = 'Female'
            elif m['is_male']: gender = 'Male'
            elif m.get('is_nonbinary'): gender = 'Non-binary'

            races = m.get('races', [])
            race_str = ', '.join(races) if races else ('White' if m['is_white'] else 'Non-White')

            round_type = team['info'].get('round_type', 'B')
            round_name = {'B': 'Both Rounds', 'C': 'First Round Only', 'D': 'Second Round Only'}.get(round_type, 'Both Rounds')

            rows.append({
                'Team ID': tid,
                'Team Day': team['info']['day'],
                'Team Time': team['info']['time'],
                'Team Format': 'Virtual' if team['info']['is_virtual'] else 'In-Person',
                'Team Round': round_name,
                'Is Facilitator': 'Yes' if role else 'No',
                'Facilitator Role': role,
                'Participant ID': m['id'],
                'Category': m.get('category', ''),
                'Year': m.get('year', ''),
                'Age Range': m.get('age_range', ''),
                'Gender': gender,
                'Race/Ethnicity': race_str,
                'Political Ideology': m.get('ideology', ''),
                'Pro Liberty Position': m.get('issue1_position', ''),
                'Pro Rule of Law Position': m.get('issue2_position', ''),
                'Format Preference': m.get('format_preference_full', ''),
                'Rounds Preference': m.get('rounds_full', ''),
                'Available Slots': m.get('total_available', 0),
                'How Heard About DCI': m.get('source', m.get('connection', '')),
                'Friend Invited': m.get('friend_invited', ''),
                'Invited By': m.get('friend_invited_by', ''),
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
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
