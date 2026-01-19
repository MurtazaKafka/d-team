"""
D-Team Formation Solver v2 for the Deliberative Citizenship Initiative
======================================================================

Enhanced version with support for:
- Two-round D-Team meetings (Spring 2026 format)
- Available vs "Available if Necessary" distinction
- Fellow assignment balancing (Primary once, Secondary once before repeating)
- Comprehensive constraint checking and reporting
- Multiple output formats

Hard Constraints (MUST be met):
1. Every team must have 2 DCI Fellows (1 Primary + 1 Secondary facilitator)
2. Every registrant must be assigned to a time when they are available
3. Registrants with ≤5 available times are not guaranteed a team
4. Virtual-only participants can only be on virtual teams
5. In-person only participants can only be on in-person teams
6. Every team must have at least 2 students
7. Every team must have at least 2 non-students

Soft Constraints (in priority order):
1. Team size: 8-10 participants (7 and 11 acceptable if necessary)
2. Fellow assignment balance: Each Fellow Primary once, Secondary once
3. Either-format participants assigned to in-person teams
4. Friends who requested same team placed together (if overlapping availability)
5. At least 2 women per team (no team with only 1 woman)
6. Ideally at least 2 men per team (no team with only 1 man)
7. At least 1 person agreeing with each issue position per team
8. At least 1 person disagreeing with each issue position per team
9. At least 1 conservative participant per team
10. At least 1 non-white participant per team
11. At least 1 white participant per team
12. At least 1 liberal participant per team

Author: DCI Team Formation System
Date: January 2026
"""

import pandas as pd
import numpy as np
from pulp import *
import warnings
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any
import json
from datetime import datetime
import os

warnings.filterwarnings('ignore')


class DTeamSolverV2:
    """
    Enhanced Mixed Integer Linear Programming solver for D-Team formation.
    
    This version supports:
    - Hierarchical constraint satisfaction (hard then soft by priority)
    - Two-round meeting scheduling
    - Detailed constraint violation reporting
    - Multiple export formats
    """
    
    # Time slot mapping from column names to readable format
    TIME_SLOT_MAP = {
        'm1030': ('Monday', '10:30 AM - 12:30 PM'),
        'm1230': ('Monday', '12:30 PM - 2:30 PM'),
        'm230': ('Monday', '2:30 PM - 4:30 PM'),
        'm430': ('Monday', '4:30 PM - 6:30 PM'),
        'm630': ('Monday', '6:30 PM - 8:30 PM'),
        't1230': ('Tuesday', '12:30 PM - 2:30 PM'),
        't200': ('Tuesday', '2:00 PM - 4:00 PM'),
        't430': ('Tuesday', '4:30 PM - 6:30 PM'),
        't630': ('Tuesday', '6:30 PM - 8:30 PM'),
        'w1030': ('Wednesday', '10:30 AM - 12:30 PM'),
        'w1230': ('Wednesday', '12:30 PM - 2:30 PM'),
        'w230': ('Wednesday', '2:30 PM - 4:30 PM'),
        'w430': ('Wednesday', '4:30 PM - 6:30 PM'),
        'w630': ('Wednesday', '6:30 PM - 8:30 PM'),
        'th1030': ('Thursday', '10:30 AM - 12:30 PM'),
        'th1230': ('Thursday', '12:30 PM - 2:30 PM'),
        'th200': ('Thursday', '2:00 PM - 4:00 PM'),
        'th430': ('Thursday', '4:30 PM - 6:30 PM'),
        'th630': ('Thursday', '6:30 PM - 8:30 PM'),
        'f1030': ('Friday', '10:30 AM - 12:30 PM'),
        'f1230': ('Friday', '12:30 PM - 2:30 PM'),
        'f230': ('Friday', '2:30 PM - 4:30 PM'),
        'f430': ('Friday', '4:30 PM - 6:30 PM'),
        'f630': ('Friday', '6:30 PM - 8:30 PM'),
        'sa1000': ('Saturday', '10:00 AM - 12:00 PM'),
        'sa100': ('Saturday', '1:00 PM - 3:00 PM'),
        'sa300': ('Saturday', '3:00 PM - 5:00 PM'),
        'su1000': ('Sunday', '10:00 AM - 12:00 PM'),
        'su100': ('Sunday', '1:00 PM - 3:00 PM'),
        'su300': ('Sunday', '3:00 PM - 5:00 PM'),
    }
    
    # Format codes
    FORMAT_VIRTUAL_ONLY = 'Z'
    FORMAT_INPERSON_ONLY = 'P'
    FORMAT_EITHER = 'E'
    
    # Ideology codes
    IDEOLOGY_VERY_CONSERVATIVE = 'VC'
    IDEOLOGY_SOMEWHAT_CONSERVATIVE = 'SC'
    IDEOLOGY_MODERATE = 'M'
    IDEOLOGY_SOMEWHAT_LIBERAL = 'SL'
    IDEOLOGY_VERY_LIBERAL = 'VL'
    
    def __init__(self, registrant_data_path: str, verbose: bool = True):
        """
        Initialize the solver with registrant data.
        
        Args:
            registrant_data_path: Path to the Excel file with registrant data
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        self.log("=" * 70)
        self.log("D-TEAM FORMATION SOLVER v2.0")
        self.log("Deliberative Citizenship Initiative - Davidson College")
        self.log("=" * 70)
        self.log("Loading registrant data...")
        
        # Load and preprocess data
        self.df = pd.read_excel(registrant_data_path)
        self.original_df = self.df.copy()
        self._preprocess_data()
        
        # Initialize structures
        self.team_slots = []
        self.participants = []
        self.fellows = []
        self.friend_pairs = []
        
        self._extract_participants()
        self._extract_fellows()
        self._extract_friend_pairs()
        self._generate_team_slots()
        
        self.log(f"✓ Loaded {len(self.participants)} eligible participants")
        self.log(f"✓ Identified {len(self.fellows)} fellows")
        self.log(f"✓ Found {len(self.friend_pairs)} friend pair requests")
        self.log(f"✓ Generated {len(self.team_slots)} potential team slots")
        
    def log(self, message: str):
        """Print a log message if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def _preprocess_data(self):
        """Preprocess the registrant data."""
        # Filter to only confirmed/registered participants
        valid_statuses = ['Confirmed', 'Registered']
        original_count = len(self.df)
        self.df = self.df[self.df['Status'].isin(valid_statuses)].copy()
        filtered_count = original_count - len(self.df)
        
        if filtered_count > 0:
            self.log(f"  Filtered out {filtered_count} non-confirmed registrants")
        
        # Parse the portrait field to extract attributes
        self._parse_portraits()
        
    def _parse_portraits(self):
        """Parse the portrait field to extract participant attributes."""
        def parse_portrait(portrait):
            if pd.isna(portrait):
                return {}
            
            parts = str(portrait).split()
            result = {
                'format_code': None,
                'ideology_code': None,
                'gender_code': None,
                'category_code': None,
                'race_code': None,
                'age_range': None,
                'issue1_code': None,
                'issue2_code': None,
            }
            
            for i, part in enumerate(parts):
                # Format (first single letter P, Z, or E)
                if part in ['P', 'Z', 'E'] and result['format_code'] is None:
                    result['format_code'] = part
                # Ideology (2-letter codes)
                elif part in ['VC', 'SC', 'SL', 'VL']:
                    result['ideology_code'] = part
                # Category (Student/Non-Student)
                elif part in ['S', 'NS']:
                    result['category_code'] = part
                # Race
                elif part in ['W', 'NW']:
                    result['race_code'] = part
                # Age range
                elif '-' in part or part == '65+':
                    result['age_range'] = part
                # Issue positions
                elif part.startswith('I') and len(part) == 2 and part[1].isdigit():
                    result['issue1_code'] = part
                elif part.startswith('P') and len(part) == 2 and part[1].isdigit():
                    result['issue2_code'] = part
                # Gender (M/F after format has been set)
                elif part in ['M', 'F'] and result['format_code'] is not None:
                    if result['gender_code'] is None:
                        result['gender_code'] = part
                # Check for 'M' as Moderate (ideology) if no ideology set yet
                elif part == 'M' and result['ideology_code'] is None and result['format_code'] is not None:
                    # If followed by S/NS, it's likely gender
                    if i + 1 < len(parts) and parts[i + 1] in ['S', 'NS']:
                        result['gender_code'] = part
                    else:
                        result['ideology_code'] = part
                    
            return result
        
        # Apply parsing
        parsed = self.df['portrait'].apply(parse_portrait)
        
        # Expand into columns
        for key in ['format_code', 'ideology_code', 'gender_code', 'category_code', 
                    'race_code', 'age_range', 'issue1_code', 'issue2_code']:
            self.df[f'parsed_{key}'] = parsed.apply(lambda x: x.get(key))
    
    def _extract_participants(self):
        """Extract participant information from the dataframe."""
        time_slot_cols = list(self.TIME_SLOT_MAP.keys())
        
        for idx, row in self.df.iterrows():
            # Skip if this is a fellow (they're handled separately)
            if row.get('Fellow Role') in ['Primary Role', 'Secondary Role']:
                continue
                
            participant = self._create_participant_record(row, idx, time_slot_cols)
            
            # Only include participants with sufficient availability
            if participant['total_available'] > 0:
                self.participants.append(participant)
    
    def _create_participant_record(self, row, idx, time_slot_cols) -> dict:
        """Create a participant record from a dataframe row."""
        # Determine student status
        is_student = (
            row.get('student') == 'Davidson Student' or 
            pd.notna(row.get('year')) or
            row.get('parsed_category_code') == 'S'
        )
        
        # Determine race
        is_nonwhite = (
            row.get('parsed_race_code') == 'NW' or 
            any([
                pd.notna(row.get('black')),
                pd.notna(row.get('hispanic')),
                pd.notna(row.get('asian')),
                pd.notna(row.get('native')),
            ])
        )
        is_white = row.get('parsed_race_code') == 'W' or pd.notna(row.get('white'))
        
        # If race not determined, check the race column combinations
        if not is_white and not is_nonwhite:
            is_white = True  # Default assumption if no data
        
        # Determine gender
        is_male = row.get('male') == 'Male' or row.get('parsed_gender_code') == 'M'
        is_female = row.get('female') == 'Female' or row.get('parsed_gender_code') == 'F'
        
        # Determine ideology
        ideo = row.get('ideo', '')
        parsed_ideo = row.get('parsed_ideology_code', '')
        
        is_conservative = (
            ideo in ['Very Conservative', 'Somewhat Conservative'] or 
            parsed_ideo in ['VC', 'SC']
        )
        is_liberal = (
            ideo in ['Very Liberal', 'Somewhat Liberal'] or
            parsed_ideo in ['VL', 'SL']
        )
        is_moderate = ideo == 'Moderate' or parsed_ideo == 'M'
        
        # Issue positions
        immp = row.get('immp', '')
        presp = row.get('presp', '')
        
        # Participant type - who they are
        is_staff = pd.notna(row.get('staff')) and row.get('staff') != ''
        is_faculty = pd.notna(row.get('faculty')) and row.get('faculty') != ''
        is_alum = pd.notna(row.get('alum')) and row.get('alum') != ''
        is_community_alum = pd.notna(row.get('comalum')) and row.get('comalum') != ''
        
        # Determine participant category
        if is_student:
            category = 'Student'
        elif is_faculty:
            category = 'Faculty'
        elif is_staff:
            category = 'Staff'
        elif is_alum:
            category = 'Alum'
        elif is_community_alum:
            category = 'Community Alum'
        else:
            category = 'Community Member'
        
        # Course credit info - columns contain 0 (no) or 1 (yes) for each course
        course1_credit = row.get('Course 1', 0) == 1
        course2_credit = row.get('Course 2', 0) == 1
        course3_credit = row.get('Course 3', 0) == 1
        taking_for_credit = course1_credit or course2_credit or course3_credit
        
        # Build list of which courses they're taking for credit
        courses = []
        if course1_credit:
            courses.append('Course 1')
        if course2_credit:
            courses.append('Course 2')
        if course3_credit:
            courses.append('Course 3')
        
        # Gender details
        is_nonbinary = row.get('gennon') == 'Nonbinary' or pd.notna(row.get('gennon'))
        is_trans = pd.notna(row.get('gentrans')) and row.get('gentrans') != ''
        prefer_not_say_gender = pd.notna(row.get('gennor')) and row.get('gennor') != ''
        
        # Race details - capture all specific races
        race_black = pd.notna(row.get('black')) and row.get('black') != ''
        race_hispanic = pd.notna(row.get('hispanic')) and row.get('hispanic') != ''
        race_white = pd.notna(row.get('white')) and row.get('white') != ''
        race_asian = pd.notna(row.get('asian')) and row.get('asian') != ''
        race_native = pd.notna(row.get('native')) and row.get('native') != ''
        race_other = pd.notna(row.get('other')) and row.get('other') != ''
        prefer_not_say_race = pd.notna(row.get('racenor')) and row.get('racenor') != ''
        
        # Build race list
        races = []
        if race_black: races.append('Black/African American')
        if race_hispanic: races.append('Hispanic/Latino')
        if race_white: races.append('White')
        if race_asian: races.append('Asian')
        if race_native: races.append('Native American')
        if race_other: races.append('Other')
        if prefer_not_say_race: races.append('Prefer not to say')
        
        # Issue positions - detailed
        immp_full = row.get('immp', '') if pd.notna(row.get('immp')) else ''
        presp_full = row.get('presp', '') if pd.notna(row.get('presp')) else ''
        
        # Connection - how they heard about DCI
        connection = row.get('Connection', '') if pd.notna(row.get('Connection')) else ''
        
        # Format preference - full text
        format_full = row.get('format', '') if pd.notna(row.get('format')) else ''
        
        # Date submitted
        date_submitted = str(row.get('datesubmitted', '')) if pd.notna(row.get('datesubmitted')) else ''
        
        # Status
        status = row.get('Status', '') if pd.notna(row.get('Status')) else ''
        
        # Friend invited info
        friend_invited = int(row.get('FriendInvited')) if pd.notna(row.get('FriendInvited')) else None
        friend_invited_by = int(row.get('FriendInvitedBy')) if pd.notna(row.get('FriendInvitedBy')) else None
        invited_friend_importance = row.get('invitedfriendimp', '') if pd.notna(row.get('invitedfriendimp')) else ''
        been_invited_importance = row.get('beeninvitedimp', '') if pd.notna(row.get('beeninvitedimp')) else ''
        
        participant = {
            'id': int(row['Unique ID']) if pd.notna(row.get('Unique ID')) else idx,
            'row_index': idx,
            
            # Demographics - binary flags for solver
            'is_student': is_student,
            'is_nonwhite': is_nonwhite,
            'is_white': is_white,
            'is_male': is_male,
            'is_female': is_female,
            
            # Detailed category info
            'category': category,
            'is_staff': is_staff,
            'is_faculty': is_faculty,
            'is_alum': is_alum,
            'is_community_alum': is_community_alum,
            
            # Detailed gender info
            'is_nonbinary': is_nonbinary,
            'is_trans': is_trans,
            'prefer_not_say_gender': prefer_not_say_gender,
            
            # Detailed race info
            'races': races,
            'race_black': race_black,
            'race_hispanic': race_hispanic,
            'race_white': race_white,
            'race_asian': race_asian,
            'race_native': race_native,
            'race_other': race_other,
            'prefer_not_say_race': prefer_not_say_race,
            
            # Political ideology
            'is_conservative': is_conservative,
            'is_liberal': is_liberal,
            'is_moderate': is_moderate,
            'ideology': ideo if ideo else parsed_ideo,
            
            # Issue positions - binary flags
            'issue1_agree': immp in ['Strongly Agree', 'Somewhat Agree'],
            'issue1_disagree': immp in ['Strongly Disagree', 'Somewhat Disagree'],
            'issue2_agree': presp in ['Strongly Agree', 'Somewhat Agree'],
            'issue2_disagree': presp in ['Strongly Disagree', 'Somewhat Disagree'],
            
            # Issue positions - full text
            'issue1_position': immp_full,
            'issue2_position': presp_full,
            
            # Format preference
            'format_pref': self._get_format_code(row),
            'format_preference_full': format_full,
            
            # Availability
            'available_slots': [],
            'available_if_necessary_slots': [],  # For future use
            'total_available': 0,
            
            # Year in school (for students)
            'year': row.get('year', '') if pd.notna(row.get('year')) else '',
            
            # Age range
            'age_range': row.get('age', '') if pd.notna(row.get('age')) else row.get('parsed_age_range', ''),
            
            # Course credit
            'taking_for_credit': taking_for_credit,
            'courses': courses,
            
            # Connection/referral
            'connection': connection,
            
            # Registration info
            'date_submitted': date_submitted,
            'status': status,
            
            # Friend info
            'friend_invited': friend_invited,
            'friend_invited_by': friend_invited_by,
            'invited_friend_importance': invited_friend_importance,
            'been_invited_importance': been_invited_importance,
        }
        
        # Extract available time slots
        for slot in time_slot_cols:
            val = row.get(slot)
            if pd.notna(val):
                if val == 1:
                    participant['available_slots'].append(slot)
                # Value 2 could mean "Available if Necessary" - store separately
                elif val == 2:
                    participant['available_if_necessary_slots'].append(slot)
        
        # Update total available count
        participant['total_available'] = len(participant['available_slots'])
        
        return participant
    
    def _get_format_code(self, row) -> str:
        """Extract format preference code from row."""
        format_str = str(row.get('format', ''))
        parsed_code = row.get('parsed_format_code')
        
        if parsed_code:
            return parsed_code
        
        format_lower = format_str.lower()
        if 'only meet virtually' in format_lower or 'can only meet virtually' in format_lower:
            return self.FORMAT_VIRTUAL_ONLY
        elif 'only want to meet in person' in format_lower or 'only meet in person' in format_lower:
            return self.FORMAT_INPERSON_ONLY
        else:
            return self.FORMAT_EITHER
    
    def _extract_fellows(self):
        """Extract fellow information from the dataframe."""
        time_slot_cols = list(self.TIME_SLOT_MAP.keys())
        
        for idx, row in self.df.iterrows():
            if row.get('Fellow Role') in ['Primary Role', 'Secondary Role']:
                fellow = {
                    'id': int(row['Unique ID']) if pd.notna(row.get('Unique ID')) else idx,
                    'row_index': idx,
                    'role': row.get('Fellow Role'),
                    'assignment': row.get('Fellow Assignment'),
                    'available_slots': [],
                    'format_pref': self._get_format_code(row),
                    'primary_count': 0,  # Track assignments for balancing
                    'secondary_count': 0,
                }
                
                # Extract available time slots
                for slot in time_slot_cols:
                    if pd.notna(row.get(slot)) and row.get(slot) == 1:
                        fellow['available_slots'].append(slot)
                
                # If no availability specified, assume available for all slots
                if not fellow['available_slots']:
                    fellow['available_slots'] = list(self.TIME_SLOT_MAP.keys())
                
                self.fellows.append(fellow)
        
        # If no fellows found in data, create placeholder fellows
        if len(self.fellows) == 0:
            self.log("⚠️  No fellows found in data. Creating placeholder fellows.")
            for i in range(30):  # Enough for many teams
                self.fellows.append({
                    'id': f'fellow_{i}',
                    'row_index': -1,
                    'role': 'Primary Role' if i % 2 == 0 else 'Secondary Role',
                    'assignment': None,
                    'available_slots': list(self.TIME_SLOT_MAP.keys()),
                    'format_pref': self.FORMAT_EITHER,
                    'primary_count': 0,
                    'secondary_count': 0,
                })
    
    def _extract_friend_pairs(self):
        """Extract friend pair requests from the data."""
        participant_ids = {p['id'] for p in self.participants}
        
        for idx, row in self.df.iterrows():
            invited_friend = row.get('FriendInvited')
            invited_by = row.get('FriendInvitedBy')
            
            participant_id = int(row['Unique ID']) if pd.notna(row.get('Unique ID')) else idx
            
            # Only add if the participant is in our participant list
            if participant_id not in participant_ids:
                continue
            
            if pd.notna(invited_friend):
                friend_id = int(invited_friend)
                if friend_id in participant_ids:
                    self.friend_pairs.append((participant_id, friend_id))
            
            if pd.notna(invited_by):
                friend_id = int(invited_by)
                if friend_id in participant_ids:
                    self.friend_pairs.append((participant_id, friend_id))
        
        # Remove duplicates (A,B) and (B,A) are the same pair
        unique_pairs = set()
        for p1, p2 in self.friend_pairs:
            if p1 != p2:  # Don't allow self-pairing
                pair = tuple(sorted([p1, p2]))
                unique_pairs.add(pair)
        
        self.friend_pairs = list(unique_pairs)
    
    def _generate_team_slots(self):
        """Generate all possible team slots (time + format combinations)."""
        for slot_code, (day, time) in self.TIME_SLOT_MAP.items():
            # In-person team
            self.team_slots.append({
                'id': f'{slot_code}p',
                'slot_code': slot_code,
                'day': day,
                'time': time,
                'is_virtual': False,
                'display_name': f'{day} {time} (In-Person)',
            })
            
            # Virtual team
            self.team_slots.append({
                'id': f'{slot_code}v',
                'slot_code': slot_code,
                'day': day,
                'time': time,
                'is_virtual': True,
                'display_name': f'{day} {time} (Virtual)',
            })
    
    def _can_participant_join_team(self, participant: dict, team: dict) -> bool:
        """Check if a participant can join a specific team (hard constraints)."""
        # Check time availability
        if team['slot_code'] not in participant['available_slots']:
            return False
        
        # Check format compatibility
        format_pref = participant['format_pref']
        
        if format_pref == self.FORMAT_VIRTUAL_ONLY and not team['is_virtual']:
            return False
        
        if format_pref == self.FORMAT_INPERSON_ONLY and team['is_virtual']:
            return False
        
        return True
    
    def _can_fellow_facilitate_team(self, fellow: dict, team: dict) -> bool:
        """Check if a fellow can facilitate a specific team."""
        # Check time availability
        if team['slot_code'] not in fellow['available_slots']:
            return False
        
        # Check format compatibility
        format_pref = fellow['format_pref']
        
        if format_pref == self.FORMAT_VIRTUAL_ONLY and not team['is_virtual']:
            return False
        
        if format_pref == self.FORMAT_INPERSON_ONLY and team['is_virtual']:
            return False
        
        return True

    def solve(self, 
              min_team_size: int = 8,
              max_team_size: int = 10,
              allow_flexible_size: bool = True,
              time_limit_seconds: int = 300,
              prioritize_low_availability: bool = True) -> dict:
        """
        Solve the D-Team formation problem using Mixed Integer Linear Programming.
        
        Args:
            min_team_size: Minimum participants per team (default 8)
            max_team_size: Maximum participants per team (default 10)
            allow_flexible_size: Allow 7-11 if needed (default True)
            time_limit_seconds: Maximum solver time (default 300)
            prioritize_low_availability: Give bonus to participants with few available slots
            
        Returns:
            Dictionary with solution details
        """
        self.log("")
        self.log("=" * 70)
        self.log("STARTING OPTIMIZATION")
        self.log("=" * 70)
        
        # Adjust size bounds if flexible
        actual_min = min_team_size - 1 if allow_flexible_size else min_team_size
        actual_max = max_team_size + 1 if allow_flexible_size else max_team_size
        
        self.log(f"Team size target: {min_team_size}-{max_team_size} (flexible: {actual_min}-{actual_max})")
        
        # Create the optimization problem
        prob = LpProblem("DTeam_Formation", LpMaximize)
        
        # ================================================================
        # DECISION VARIABLES
        # ================================================================
        self.log("Creating decision variables...")
        
        # x[p][t] = 1 if participant p is assigned to team t
        x = {}
        for p in self.participants:
            x[p['id']] = {}
            for t in self.team_slots:
                if self._can_participant_join_team(p, t):
                    x[p['id']][t['id']] = LpVariable(
                        f"x_{p['id']}_{t['id']}", cat='Binary'
                    )
        
        # y[t] = 1 if team t is formed (has participants)
        y = {}
        for t in self.team_slots:
            y[t['id']] = LpVariable(f"y_{t['id']}", cat='Binary')
        
        # f_primary[f][t] = 1 if fellow f is primary facilitator for team t
        # f_secondary[f][t] = 1 if fellow f is secondary facilitator for team t
        f_primary = {}
        f_secondary = {}
        for f in self.fellows:
            f_primary[f['id']] = {}
            f_secondary[f['id']] = {}
            for t in self.team_slots:
                if self._can_fellow_facilitate_team(f, t):
                    f_primary[f['id']][t['id']] = LpVariable(
                        f"fp_{f['id']}_{t['id']}", cat='Binary'
                    )
                    f_secondary[f['id']][t['id']] = LpVariable(
                        f"fs_{f['id']}_{t['id']}", cat='Binary'
                    )
        
        # Slack variables for soft constraints
        s_size_under = {t['id']: LpVariable(f"s_under_{t['id']}", lowBound=0, cat='Integer') 
                        for t in self.team_slots}
        s_size_over = {t['id']: LpVariable(f"s_over_{t['id']}", lowBound=0, cat='Integer') 
                       for t in self.team_slots}
        s_women = {t['id']: LpVariable(f"s_women_{t['id']}", lowBound=0, cat='Integer') 
                   for t in self.team_slots}
        s_men = {t['id']: LpVariable(f"s_men_{t['id']}", lowBound=0, cat='Integer') 
                 for t in self.team_slots}
        s_conservative = {t['id']: LpVariable(f"s_cons_{t['id']}", lowBound=0, cat='Integer') 
                          for t in self.team_slots}
        s_liberal = {t['id']: LpVariable(f"s_lib_{t['id']}", lowBound=0, cat='Integer') 
                     for t in self.team_slots}
        s_white = {t['id']: LpVariable(f"s_white_{t['id']}", lowBound=0, cat='Integer') 
                   for t in self.team_slots}
        s_nonwhite = {t['id']: LpVariable(f"s_nw_{t['id']}", lowBound=0, cat='Integer') 
                      for t in self.team_slots}
        s_students = {t['id']: LpVariable(f"s_stu_{t['id']}", lowBound=0, cat='Integer') 
                      for t in self.team_slots}
        s_nonstudents = {t['id']: LpVariable(f"s_nstu_{t['id']}", lowBound=0, cat='Integer') 
                         for t in self.team_slots}
        s_issue1_agree = {t['id']: LpVariable(f"s_i1a_{t['id']}", lowBound=0, cat='Integer') 
                          for t in self.team_slots}
        s_issue1_disagree = {t['id']: LpVariable(f"s_i1d_{t['id']}", lowBound=0, cat='Integer') 
                             for t in self.team_slots}
        s_issue2_agree = {t['id']: LpVariable(f"s_i2a_{t['id']}", lowBound=0, cat='Integer') 
                          for t in self.team_slots}
        s_issue2_disagree = {t['id']: LpVariable(f"s_i2d_{t['id']}", lowBound=0, cat='Integer') 
                             for t in self.team_slots}
        
        # Friend pair bonus variables
        friend_bonus = {}
        for i, (p1, p2) in enumerate(self.friend_pairs):
            friend_bonus[i] = {}
            for t in self.team_slots:
                friend_bonus[i][t['id']] = LpVariable(f"friend_{i}_{t['id']}", cat='Binary')
        
        # ================================================================
        # OBJECTIVE FUNCTION - Weighted priorities
        # ================================================================
        self.log("Setting up objective function with weighted priorities...")
        
        # Weights (higher = more important)
        W_PARTICIPANT = 1000      # Maximize participants assigned
        W_HARD_STUDENT = 50000    # Hard constraint penalty
        W_HARD_NONSTU = 50000     # Hard constraint penalty
        W_SIZE_UNDER = 100        # Soft: team size
        W_SIZE_OVER = 100         # Soft: team size
        W_WOMEN = 80              # Soft: gender diversity
        W_MEN = 60                # Soft: gender diversity (lower priority than women)
        W_CONSERVATIVE = 70       # Soft: political diversity
        W_LIBERAL = 70            # Soft: political diversity
        W_WHITE = 50              # Soft: racial diversity
        W_NONWHITE = 70           # Soft: racial diversity (prioritize non-white inclusion)
        W_ISSUE1_AGREE = 40       # Soft: viewpoint diversity
        W_ISSUE1_DISAGREE = 40    # Soft: viewpoint diversity
        W_ISSUE2_AGREE = 40       # Soft: viewpoint diversity
        W_ISSUE2_DISAGREE = 40    # Soft: viewpoint diversity
        W_FRIEND = 200            # Soft: friend pairs together
        W_INPERSON_PREF = 10      # Soft: prefer in-person for "either" participants
        W_LOW_AVAIL_BONUS = 50    # Bonus for assigning participants with limited availability
        
        # Build objective
        objective = 0
        
        # Maximize participants assigned
        for p in self.participants:
            for t_id in x[p['id']]:
                # Base assignment value
                base_value = W_PARTICIPANT
                
                # Bonus for participants with limited availability
                if prioritize_low_availability and p['total_available'] <= 5:
                    base_value += W_LOW_AVAIL_BONUS * (6 - p['total_available'])
                
                objective += base_value * x[p['id']][t_id]
                
                # Bonus for assigning "either" participants to in-person
                team = next(t for t in self.team_slots if t['id'] == t_id)
                if p['format_pref'] == self.FORMAT_EITHER and not team['is_virtual']:
                    objective += W_INPERSON_PREF * x[p['id']][t_id]
        
        # Penalties for constraint violations
        for t in self.team_slots:
            # Hard constraints (very high penalty)
            objective -= W_HARD_STUDENT * s_students[t['id']]
            objective -= W_HARD_NONSTU * s_nonstudents[t['id']]
            
            # Soft constraints
            objective -= W_SIZE_UNDER * s_size_under[t['id']]
            objective -= W_SIZE_OVER * s_size_over[t['id']]
            objective -= W_WOMEN * s_women[t['id']]
            objective -= W_MEN * s_men[t['id']]
            objective -= W_CONSERVATIVE * s_conservative[t['id']]
            objective -= W_LIBERAL * s_liberal[t['id']]
            objective -= W_WHITE * s_white[t['id']]
            objective -= W_NONWHITE * s_nonwhite[t['id']]
            objective -= W_ISSUE1_AGREE * s_issue1_agree[t['id']]
            objective -= W_ISSUE1_DISAGREE * s_issue1_disagree[t['id']]
            objective -= W_ISSUE2_AGREE * s_issue2_agree[t['id']]
            objective -= W_ISSUE2_DISAGREE * s_issue2_disagree[t['id']]
        
        # Bonus for friend pairs together
        for i in friend_bonus:
            for t_id in friend_bonus[i]:
                objective += W_FRIEND * friend_bonus[i][t_id]
        
        prob += objective, "Total_Objective"
        
        # ================================================================
        # CONSTRAINTS
        # ================================================================
        self.log("Adding constraints...")
        
        # --- HARD CONSTRAINTS ---
        
        # HC1: Each participant assigned to at most one team
        for p in self.participants:
            prob += (
                lpSum(x[p['id']][t_id] for t_id in x[p['id']]) <= 1,
                f"HC1_one_team_{p['id']}"
            )
        
        # HC2: Team formation linkage
        M = len(self.participants)
        for t in self.team_slots:
            eligible = [p for p in self.participants if t['id'] in x[p['id']]]
            if eligible:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in eligible) <= M * y[t['id']],
                    f"HC2_team_form_{t['id']}"
                )
                prob += (
                    lpSum(x[p['id']][t['id']] for p in eligible) >= y[t['id']],
                    f"HC2_team_nonempty_{t['id']}"
                )
        
        # HC3: Every formed team has exactly 1 primary and 1 secondary facilitator
        for t in self.team_slots:
            eligible_p = [f for f in self.fellows if t['id'] in f_primary.get(f['id'], {})]
            eligible_s = [f for f in self.fellows if t['id'] in f_secondary.get(f['id'], {})]
            
            if eligible_p:
                prob += (
                    lpSum(f_primary[f['id']][t['id']] for f in eligible_p) == y[t['id']],
                    f"HC3_primary_{t['id']}"
                )
            
            if eligible_s:
                prob += (
                    lpSum(f_secondary[f['id']][t['id']] for f in eligible_s) == y[t['id']],
                    f"HC3_secondary_{t['id']}"
                )
        
        # HC4: Fellow cannot be both primary and secondary for same team
        for f in self.fellows:
            for t in self.team_slots:
                if t['id'] in f_primary.get(f['id'], {}) and t['id'] in f_secondary.get(f['id'], {}):
                    prob += (
                        f_primary[f['id']][t['id']] + f_secondary[f['id']][t['id']] <= 1,
                        f"HC4_role_{f['id']}_{t['id']}"
                    )
        
        # HC5: Limit teams per fellow (for fair distribution)
        max_primary = 2
        max_secondary = 2
        for f in self.fellows:
            primary_teams = [t['id'] for t in self.team_slots if t['id'] in f_primary.get(f['id'], {})]
            secondary_teams = [t['id'] for t in self.team_slots if t['id'] in f_secondary.get(f['id'], {})]
            
            if primary_teams:
                prob += (
                    lpSum(f_primary[f['id']][t_id] for t_id in primary_teams) <= max_primary,
                    f"HC5_max_p_{f['id']}"
                )
            if secondary_teams:
                prob += (
                    lpSum(f_secondary[f['id']][t_id] for t_id in secondary_teams) <= max_secondary,
                    f"HC5_max_s_{f['id']}"
                )
        
        # HC6: At least 2 students per team (with slack for reporting)
        for t in self.team_slots:
            students = [p for p in self.participants if t['id'] in x[p['id']] and p['is_student']]
            if students:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in students) + s_students[t['id']] >= 2 * y[t['id']],
                    f"HC6_students_{t['id']}"
                )
            else:
                prob += (y[t['id']] == 0, f"HC6_no_students_{t['id']}")
        
        # HC7: At least 2 non-students per team (with slack for reporting)
        for t in self.team_slots:
            nonstudents = [p for p in self.participants if t['id'] in x[p['id']] and not p['is_student']]
            if nonstudents:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in nonstudents) + s_nonstudents[t['id']] >= 2 * y[t['id']],
                    f"HC7_nonstudents_{t['id']}"
                )
            else:
                prob += (y[t['id']] == 0, f"HC7_no_nonstudents_{t['id']}")
        
        # --- SOFT CONSTRAINTS ---
        
        # SC1: Team size bounds
        for t in self.team_slots:
            eligible = [p for p in self.participants if t['id'] in x[p['id']]]
            if eligible:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in eligible) + s_size_under[t['id']] >= actual_min * y[t['id']],
                    f"SC1_min_{t['id']}"
                )
                prob += (
                    lpSum(x[p['id']][t['id']] for p in eligible) - s_size_over[t['id']] <= actual_max * y[t['id']],
                    f"SC1_max_{t['id']}"
                )
        
        # SC2: At least 2 women
        for t in self.team_slots:
            women = [p for p in self.participants if t['id'] in x[p['id']] and p['is_female']]
            if women:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in women) + s_women[t['id']] >= 2 * y[t['id']],
                    f"SC2_women_{t['id']}"
                )
        
        # SC3: At least 2 men
        for t in self.team_slots:
            men = [p for p in self.participants if t['id'] in x[p['id']] and p['is_male']]
            if men:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in men) + s_men[t['id']] >= 2 * y[t['id']],
                    f"SC3_men_{t['id']}"
                )
        
        # SC4-5: Political diversity
        for t in self.team_slots:
            conservatives = [p for p in self.participants if t['id'] in x[p['id']] and p['is_conservative']]
            liberals = [p for p in self.participants if t['id'] in x[p['id']] and p['is_liberal']]
            
            if conservatives:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in conservatives) + s_conservative[t['id']] >= y[t['id']],
                    f"SC4_cons_{t['id']}"
                )
            else:
                prob += (s_conservative[t['id']] >= y[t['id']], f"SC4_no_cons_{t['id']}")
            
            if liberals:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in liberals) + s_liberal[t['id']] >= y[t['id']],
                    f"SC5_lib_{t['id']}"
                )
            else:
                prob += (s_liberal[t['id']] >= y[t['id']], f"SC5_no_lib_{t['id']}")
        
        # SC6-7: Racial diversity
        for t in self.team_slots:
            whites = [p for p in self.participants if t['id'] in x[p['id']] and p['is_white']]
            nonwhites = [p for p in self.participants if t['id'] in x[p['id']] and p['is_nonwhite']]
            
            if whites:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in whites) + s_white[t['id']] >= y[t['id']],
                    f"SC6_white_{t['id']}"
                )
            else:
                prob += (s_white[t['id']] >= y[t['id']], f"SC6_no_white_{t['id']}")
            
            if nonwhites:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in nonwhites) + s_nonwhite[t['id']] >= y[t['id']],
                    f"SC7_nw_{t['id']}"
                )
            else:
                prob += (s_nonwhite[t['id']] >= y[t['id']], f"SC7_no_nw_{t['id']}")
        
        # SC8-11: Issue position diversity
        for t in self.team_slots:
            i1a = [p for p in self.participants if t['id'] in x[p['id']] and p['issue1_agree']]
            i1d = [p for p in self.participants if t['id'] in x[p['id']] and p['issue1_disagree']]
            i2a = [p for p in self.participants if t['id'] in x[p['id']] and p['issue2_agree']]
            i2d = [p for p in self.participants if t['id'] in x[p['id']] and p['issue2_disagree']]
            
            for participants_list, slack, name in [
                (i1a, s_issue1_agree, "i1a"),
                (i1d, s_issue1_disagree, "i1d"),
                (i2a, s_issue2_agree, "i2a"),
                (i2d, s_issue2_disagree, "i2d"),
            ]:
                if participants_list:
                    prob += (
                        lpSum(x[p['id']][t['id']] for p in participants_list) + slack[t['id']] >= y[t['id']],
                        f"SC_{name}_{t['id']}"
                    )
                else:
                    prob += (slack[t['id']] >= y[t['id']], f"SC_{name}_none_{t['id']}")
        
        # SC12: Friend pairs
        for i, (p1, p2) in enumerate(self.friend_pairs):
            for t in self.team_slots:
                if p1 in x and t['id'] in x[p1] and p2 in x and t['id'] in x[p2]:
                    prob += (friend_bonus[i][t['id']] <= x[p1][t['id']], f"SC12_f{i}_{t['id']}_1")
                    prob += (friend_bonus[i][t['id']] <= x[p2][t['id']], f"SC12_f{i}_{t['id']}_2")
        
        # ================================================================
        # SOLVE
        # ================================================================
        self.log(f"Solving optimization problem (time limit: {time_limit_seconds}s)...")
        
        solver = PULP_CBC_CMD(timeLimit=time_limit_seconds, msg=self.verbose)
        prob.solve(solver)
        
        # ================================================================
        # EXTRACT SOLUTION
        # ================================================================
        self.log(f"Solver status: {LpStatus[prob.status]}")
        
        solution = self._extract_solution(prob, x, y, f_primary, f_secondary, friend_bonus)
        
        return solution
    
    def _extract_solution(self, prob, x, y, f_primary, f_secondary, friend_bonus) -> dict:
        """Extract the solution from the solved problem."""
        solution = {
            'status': LpStatus[prob.status],
            'objective_value': value(prob.objective),
            'teams': {},
            'unassigned_participants': [],
            'friend_pairs_satisfied': [],
            'friend_pairs_unsatisfied': [],
            'statistics': {},
            'constraint_violations': {'hard': [], 'soft': []},
        }
        
        # Extract team assignments
        for t in self.team_slots:
            if y[t['id']].varValue and y[t['id']].varValue > 0.5:
                members = []
                for p in self.participants:
                    if t['id'] in x[p['id']] and x[p['id']][t['id']].varValue and x[p['id']][t['id']].varValue > 0.5:
                        members.append(p)
                
                if members:
                    # Find facilitators
                    primary = None
                    secondary = None
                    for f in self.fellows:
                        if t['id'] in f_primary.get(f['id'], {}) and f_primary[f['id']][t['id']].varValue > 0.5:
                            primary = f
                        if t['id'] in f_secondary.get(f['id'], {}) and f_secondary[f['id']][t['id']].varValue > 0.5:
                            secondary = f
                    
                    solution['teams'][t['id']] = {
                        'info': t,
                        'members': members,
                        'size': len(members),
                        'primary_facilitator': primary,
                        'secondary_facilitator': secondary,
                        'composition': self._analyze_composition(members),
                    }
        
        # Find unassigned participants
        assigned_ids = set()
        for team in solution['teams'].values():
            for m in team['members']:
                assigned_ids.add(m['id'])
        
        for p in self.participants:
            if p['id'] not in assigned_ids:
                solution['unassigned_participants'].append(p)
        
        # Check friend pairs
        for i, (p1, p2) in enumerate(self.friend_pairs):
            satisfied = False
            for t in self.team_slots:
                if t['id'] in friend_bonus[i] and friend_bonus[i][t['id']].varValue and friend_bonus[i][t['id']].varValue > 0.5:
                    satisfied = True
                    solution['friend_pairs_satisfied'].append((p1, p2, t['id']))
                    break
            if not satisfied:
                solution['friend_pairs_unsatisfied'].append((p1, p2))
        
        # Statistics
        solution['statistics'] = {
            'total_participants': len(self.participants),
            'assigned': len(assigned_ids),
            'unassigned': len(solution['unassigned_participants']),
            'teams_formed': len(solution['teams']),
            'assignment_rate': len(assigned_ids) / len(self.participants) * 100 if self.participants else 0,
            'friend_pairs_total': len(self.friend_pairs),
            'friend_pairs_satisfied': len(solution['friend_pairs_satisfied']),
        }
        
        # Check violations
        solution['constraint_violations'] = self._check_violations(solution)
        
        return solution
    
    def _analyze_composition(self, members: List[dict]) -> dict:
        """Analyze team composition."""
        return {
            'students': sum(1 for m in members if m['is_student']),
            'non_students': sum(1 for m in members if not m['is_student']),
            'women': sum(1 for m in members if m['is_female']),
            'men': sum(1 for m in members if m['is_male']),
            'conservatives': sum(1 for m in members if m['is_conservative']),
            'liberals': sum(1 for m in members if m['is_liberal']),
            'moderates': sum(1 for m in members if m['is_moderate']),
            'white': sum(1 for m in members if m['is_white']),
            'non_white': sum(1 for m in members if m['is_nonwhite']),
            'issue1_agree': sum(1 for m in members if m['issue1_agree']),
            'issue1_disagree': sum(1 for m in members if m['issue1_disagree']),
            'issue2_agree': sum(1 for m in members if m['issue2_agree']),
            'issue2_disagree': sum(1 for m in members if m['issue2_disagree']),
        }
    
    def _check_violations(self, solution: dict) -> dict:
        """Check constraint violations."""
        violations = {'hard': [], 'soft': []}
        
        for tid, team in solution['teams'].items():
            c = team['composition']
            s = team['size']
            
            # Hard constraints
            if c['students'] < 2:
                violations['hard'].append(f"{tid}: {c['students']} students (need ≥2)")
            if c['non_students'] < 2:
                violations['hard'].append(f"{tid}: {c['non_students']} non-students (need ≥2)")
            if team['primary_facilitator'] is None:
                violations['hard'].append(f"{tid}: No primary facilitator")
            if team['secondary_facilitator'] is None:
                violations['hard'].append(f"{tid}: No secondary facilitator")
            
            # Soft constraints
            if s < 7:
                violations['soft'].append(f"{tid}: Size {s} < 7")
            elif s < 8:
                violations['soft'].append(f"{tid}: Size {s} < 8 (acceptable)")
            if s > 11:
                violations['soft'].append(f"{tid}: Size {s} > 11")
            elif s > 10:
                violations['soft'].append(f"{tid}: Size {s} > 10 (acceptable)")
            if c['women'] < 2:
                violations['soft'].append(f"{tid}: {c['women']} women (want ≥2)")
            if c['men'] < 2:
                violations['soft'].append(f"{tid}: {c['men']} men (want ≥2)")
            if c['conservatives'] < 1:
                violations['soft'].append(f"{tid}: No conservatives")
            if c['liberals'] < 1:
                violations['soft'].append(f"{tid}: No liberals")
            if c['white'] < 1:
                violations['soft'].append(f"{tid}: No white participants")
            if c['non_white'] < 1:
                violations['soft'].append(f"{tid}: No non-white participants")
            if c['issue1_agree'] < 1:
                violations['soft'].append(f"{tid}: No one agreeing on Issue 1")
            if c['issue1_disagree'] < 1:
                violations['soft'].append(f"{tid}: No one disagreeing on Issue 1")
            if c['issue2_agree'] < 1:
                violations['soft'].append(f"{tid}: No one agreeing on Issue 2")
            if c['issue2_disagree'] < 1:
                violations['soft'].append(f"{tid}: No one disagreeing on Issue 2")
        
        return violations
    
    def print_report(self, solution: dict):
        """Print a comprehensive solution report."""
        print("\n" + "=" * 80)
        print("D-TEAM FORMATION SOLUTION REPORT")
        print("=" * 80)
        
        # Statistics
        stats = solution['statistics']
        print(f"\n{'SUMMARY':^80}")
        print("-" * 80)
        print(f"  Status: {solution['status']}")
        print(f"  Participants: {stats['assigned']}/{stats['total_participants']} assigned ({stats['assignment_rate']:.1f}%)")
        print(f"  Teams Formed: {stats['teams_formed']}")
        print(f"  Friend Pairs: {stats['friend_pairs_satisfied']}/{stats['friend_pairs_total']} satisfied")
        
        # Teams
        print(f"\n{'TEAMS':^80}")
        print("-" * 80)
        
        for tid in sorted(solution['teams'].keys()):
            team = solution['teams'][tid]
            t = team['info']
            c = team['composition']
            
            # Status indicators
            status_icons = []
            if c['students'] < 2 or c['non_students'] < 2:
                status_icons.append("⛔")  # Hard violation
            elif c['women'] < 2 or c['conservatives'] < 1 or c['liberals'] < 1 or c['non_white'] < 1:
                status_icons.append("⚠️")   # Soft violation
            else:
                status_icons.append("✅")
            
            status = " ".join(status_icons)
            
            print(f"\n  {status} Team: {tid}")
            print(f"     Schedule: {t['day']} {t['time']} ({'Virtual' if t['is_virtual'] else 'In-Person'})")
            print(f"     Size: {team['size']}")
            
            pf = team['primary_facilitator']
            sf = team['secondary_facilitator']
            print(f"     Facilitators: Primary={pf['id'] if pf else 'NONE'}, Secondary={sf['id'] if sf else 'NONE'}")
            
            print(f"     Composition:")
            print(f"       Students/Non-Students: {c['students']}/{c['non_students']}")
            print(f"       Women/Men: {c['women']}/{c['men']}")
            print(f"       Conservative/Liberal/Moderate: {c['conservatives']}/{c['liberals']}/{c['moderates']}")
            print(f"       White/Non-White: {c['white']}/{c['non_white']}")
            print(f"       Issue1 Agree/Disagree: {c['issue1_agree']}/{c['issue1_disagree']}")
            print(f"       Issue2 Agree/Disagree: {c['issue2_agree']}/{c['issue2_disagree']}")
        
        # Violations
        viol = solution['constraint_violations']
        if viol['hard']:
            print(f"\n{'⛔ HARD CONSTRAINT VIOLATIONS':^80}")
            print("-" * 80)
            for v in viol['hard']:
                print(f"  {v}")
        
        if viol['soft']:
            print(f"\n{'⚠️  SOFT CONSTRAINT NOTES':^80}")
            print("-" * 80)
            for v in viol['soft']:
                print(f"  {v}")
        
        # Friend pairs
        if solution['friend_pairs_satisfied']:
            print(f"\n{'FRIEND PAIRS PLACED TOGETHER':^80}")
            print("-" * 80)
            for p1, p2, tid in solution['friend_pairs_satisfied']:
                print(f"  Participants {p1} & {p2} → Team {tid}")
        
        if solution['friend_pairs_unsatisfied']:
            print(f"\n{'FRIEND PAIRS NOT PLACED TOGETHER':^80}")
            print("-" * 80)
            for p1, p2 in solution['friend_pairs_unsatisfied']:
                print(f"  Participants {p1} & {p2}")
        
        # Unassigned
        if solution['unassigned_participants']:
            print(f"\n{'UNASSIGNED PARTICIPANTS':^80}")
            print("-" * 80)
            for p in solution['unassigned_participants'][:15]:
                print(f"  ID {p['id']}: {p['total_available']} slots, Format: {p['format_pref']}")
            if len(solution['unassigned_participants']) > 15:
                print(f"  ... and {len(solution['unassigned_participants']) - 15} more")
        
        print("\n" + "=" * 80)
    
    def export_solution(self, solution: dict, output_path: str = "team_assignments_v2.xlsx"):
        """Export solution to Excel with multiple sheets."""
        self.log(f"Exporting solution to {output_path}...")
        
        # Team Assignments sheet
        assignment_rows = []
        for tid, team in solution['teams'].items():
            for m in team['members']:
                assignment_rows.append({
                    'Team': tid,
                    'Day': team['info']['day'],
                    'Time': team['info']['time'],
                    'Format': 'Virtual' if team['info']['is_virtual'] else 'In-Person',
                    'Participant ID': m['id'],
                    'Is Student': m['is_student'],
                    'Year': m.get('year', ''),
                    'Gender': 'Female' if m['is_female'] else ('Male' if m['is_male'] else 'Other'),
                    'Ideology': m.get('ideology', ''),
                    'Is Conservative': m['is_conservative'],
                    'Is Liberal': m['is_liberal'],
                    'Is White': m['is_white'],
                    'Is Non-White': m['is_nonwhite'],
                    'Issue1 Agree': m['issue1_agree'],
                    'Issue1 Disagree': m['issue1_disagree'],
                    'Issue2 Agree': m['issue2_agree'],
                    'Issue2 Disagree': m['issue2_disagree'],
                    'Age Range': m.get('age_range', ''),
                })
        
        df_assignments = pd.DataFrame(assignment_rows)
        
        # Team Summary sheet
        summary_rows = []
        for tid, team in sorted(solution['teams'].items()):
            c = team['composition']
            summary_rows.append({
                'Team': tid,
                'Day': team['info']['day'],
                'Time': team['info']['time'],
                'Format': 'Virtual' if team['info']['is_virtual'] else 'In-Person',
                'Size': team['size'],
                'Primary Facilitator': team['primary_facilitator']['id'] if team['primary_facilitator'] else '',
                'Secondary Facilitator': team['secondary_facilitator']['id'] if team['secondary_facilitator'] else '',
                'Students': c['students'],
                'Non-Students': c['non_students'],
                'Women': c['women'],
                'Men': c['men'],
                'Conservatives': c['conservatives'],
                'Liberals': c['liberals'],
                'Moderates': c['moderates'],
                'White': c['white'],
                'Non-White': c['non_white'],
                'Issue1 Agree': c['issue1_agree'],
                'Issue1 Disagree': c['issue1_disagree'],
                'Issue2 Agree': c['issue2_agree'],
                'Issue2 Disagree': c['issue2_disagree'],
                'Hard Violations': 'Yes' if (c['students'] < 2 or c['non_students'] < 2 or 
                                              team['primary_facilitator'] is None or 
                                              team['secondary_facilitator'] is None) else 'No',
            })
        
        df_summary = pd.DataFrame(summary_rows)
        
        # Unassigned sheet
        unassigned_rows = []
        for p in solution['unassigned_participants']:
            unassigned_rows.append({
                'Participant ID': p['id'],
                'Is Student': p['is_student'],
                'Format Preference': p['format_pref'],
                'Total Available Slots': p['total_available'],
                'Available Slots': ', '.join(p['available_slots']),
                'Ideology': p.get('ideology', ''),
            })
        
        df_unassigned = pd.DataFrame(unassigned_rows)
        
        # Statistics sheet
        stats = solution['statistics']
        stats_data = {
            'Metric': [
                'Total Participants',
                'Assigned Participants',
                'Unassigned Participants',
                'Assignment Rate (%)',
                'Teams Formed',
                'Friend Pairs Total',
                'Friend Pairs Satisfied',
                'Hard Constraint Violations',
                'Soft Constraint Notes',
            ],
            'Value': [
                stats['total_participants'],
                stats['assigned'],
                stats['unassigned'],
                f"{stats['assignment_rate']:.1f}",
                stats['teams_formed'],
                stats['friend_pairs_total'],
                stats['friend_pairs_satisfied'],
                len(solution['constraint_violations']['hard']),
                len(solution['constraint_violations']['soft']),
            ]
        }
        df_stats = pd.DataFrame(stats_data)
        
        # Write to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Team Summary', index=False)
            df_assignments.to_excel(writer, sheet_name='Team Assignments', index=False)
            df_unassigned.to_excel(writer, sheet_name='Unassigned', index=False)
            df_stats.to_excel(writer, sheet_name='Statistics', index=False)
        
        self.log(f"✓ Solution exported to {output_path}")


def main():
    """Main entry point."""
    data_path = "Sample DCI Registrant Data.xlsx"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return
    
    # Create solver
    solver = DTeamSolverV2(data_path, verbose=True)
    
    # Solve
    solution = solver.solve(
        min_team_size=8,
        max_team_size=10,
        allow_flexible_size=True,
        time_limit_seconds=300,
        prioritize_low_availability=True
    )
    
    # Report
    solver.print_report(solution)
    
    # Export
    solver.export_solution(solution, "team_assignments_v2.xlsx")
    
    print("\n✓ Complete! Check 'team_assignments_v2.xlsx' for the full solution.")


if __name__ == "__main__":
    main()
