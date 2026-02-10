"""
D-Team Formation Solver v3 for the Deliberative Citizenship Initiative
======================================================================

Version 3 - Qualtrics Data Format Support (Spring 2026)

Key changes from v2:
- Supports Qualtrics CSV export format (skips first 2 header rows)
- Also supports Qualtrics Excel export format
- Three availability sets based on round preference (Both, First Only, Second Only)
- Round-based team formation (teams meet once or twice)
- New availability values: 1=Available, 2=Available if Necessary, 3=Unavailable
- New soft constraint: prefer "Available" over "Available if Necessary"
- New issue columns: Pro Liberty and Pro Rule of Law
- Real DCI Fellows extracted from data (Facilitator column)
- Mutual friend consent required for friend pairing
- Prioritize fellows who have NOT been facilitator before

Hard Constraints (MUST be met):
1. Every team must have 2 DCI Fellows (1 Primary + 1 Secondary facilitator)
2. No one assigned to unavailable times (value=3)
3. Virtual-only participants can only be on virtual teams
4. In-person only participants can only be on in-person teams
5. Every team must have at least 2 students
6. Every team must have at least 2 non-students
7. Round matching: First-round-only participants in first-round teams, etc.

Soft Constraints (in priority order):
1. Team size: 8-10 participants (7 and 11 acceptable if necessary)
2. Fellow assignment balance: Each Fellow Primary once, Secondary once
3. Either-format participants assigned to in-person teams
4. Friends who mutually requested same team placed together
5. At least 2 women per team
6. Ideally at least 2 men per team
7. At least 1 person agreeing with Pro Liberty per team
8. At least 1 person disagreeing with Pro Liberty per team
9. At least 1 person agreeing with Pro Rule of Law per team
10. At least 1 person disagreeing with Pro Rule of Law per team
11. At least 1 conservative participant per team
12. At least 1 non-white participant per team
13. At least 1 white participant per team
14. At least 1 liberal participant per team
15. Prefer "Available" times over "Available if Necessary" times (lowest priority)

Author: DCI Team Formation System
Date: February 2026
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
import re

warnings.filterwarnings('ignore')


class DTeamSolverV3:
    """
    Mixed Integer Linear Programming solver for D-Team formation.
    Supports Qualtrics data format for Spring 2026.
    """

    # Time slot mapping for Qualtrics format
    TIME_SLOT_MAP_QUALTRICS = {
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
        'm1030': ('Monday', '10:30 AM - 12:30 PM'),
        'm1230': ('Monday', '12:30 PM - 2:30 PM'),
        'm230': ('Monday', '2:30 PM - 4:30 PM'),
        'm430': ('Monday', '4:30 PM - 6:30 PM'),
        'm630': ('Monday', '6:30 PM - 8:30 PM'),
        't1000': ('Tuesday', '10:00 AM - 12:00 PM'),
        't1100': ('Tuesday', '11:00 AM - 1:00 PM'),
        't1230': ('Tuesday', '12:30 PM - 2:30 PM'),
        't330': ('Tuesday', '3:30 PM - 5:30 PM'),
        't430': ('Tuesday', '4:30 PM - 6:30 PM'),
        't630': ('Tuesday', '6:30 PM - 8:30 PM'),
        'w1030': ('Wednesday', '10:30 AM - 12:30 PM'),
        'w1230': ('Wednesday', '12:30 PM - 2:30 PM'),
        'w230': ('Wednesday', '2:30 PM - 4:30 PM'),
        'w430': ('Wednesday', '4:30 PM - 6:30 PM'),
        'w630': ('Wednesday', '6:30 PM - 8:30 PM'),
        'th1230': ('Thursday', '12:30 PM - 2:30 PM'),
        'th330': ('Thursday', '3:30 PM - 5:30 PM'),
        'th430': ('Thursday', '4:30 PM - 6:30 PM'),
        'th630': ('Thursday', '6:30 PM - 8:30 PM'),
    }

    # Mapping from Qualtrics column names to internal slot codes
    # The CSV data row 3 (short variable names) uses these column headers:
    #   Both rounds: 2f1030, 2f1230 ... (prefix "2" for Both)
    #   First round: C2f1030, C2f1230 ... (prefix "C2")
    #   Second round: B2f1030, B2f1230 ... (prefix "B2")
    #   Thursday 2nd round: Q48_1..Q48_4
    # But the actual column headers in row 0 of the CSV are Fri_1, Sat_1, etc.
    # We use the row 0 column headers as that's what pandas uses.
    QUALTRICS_COL_MAP = {
        # Both rounds - Fri_ columns
        'Fri_1': 'Bf1030', 'Fri_2': 'Bf1230', 'Fri_3': 'Bf230', 'Fri_4': 'Bf430', 'Fri_5': 'Bf630',
        'Sat_1': 'Bsa1000', 'Sat_2': 'Bsa100', 'Sat_3': 'Bsa300',
        'Sun_1': 'Bsu1000', 'Sun_2': 'Bsu100', 'Sun_3': 'Bsu300',
        'Mon_1': 'Bm1030', 'Mon_2': 'Bm1230', 'Mon_3': 'Bm230', 'Mon_4': 'Bm430', 'Mon_5': 'Bm630',
        'Tue_1': 'Bt1000', 'Tue_2': 'Bt1100', 'Tue_3': 'Bt1230', 'Tue_4': 'Bt330', 'Tue_5': 'Bt430', 'Tue_6': 'Bt630',
        'Wed_1': 'Bw1030', 'Wed_2': 'Bw1230', 'Wed_3': 'Bw230', 'Wed_4': 'Bw430', 'Wed_5': 'Bw630',
        'Thu_1': 'Bth1230', 'Thu_2': 'Bth330', 'Thu_3': 'Bth430', 'Thu_4': 'Bth630',

        # First round only - Fri1_ columns
        'Fri1_1': 'Cf1030', 'Fri1_2': 'Cf1230', 'Fri1_3': 'Cf230', 'Fri1_4': 'Cf430', 'Fri1_5': 'Cf630',
        'Sat1_1': 'Csa1000', 'Sat1_2': 'Csa100', 'Sat1_3': 'Csa300',
        'Sun1_1': 'Csu1000', 'Sun1_2': 'Csu100', 'Sun1_3': 'Csu300',
        'Mon1_1': 'Cm1030', 'Mon1_2': 'Cm1230', 'Mon1_3': 'Cm230', 'Mon1_4': 'Cm430', 'Mon1_5': 'Cm630',
        'Tue1_1': 'Ct1000', 'Tue1_2': 'Ct1100', 'Tue1_3': 'Ct1230', 'Tue1_4': 'Ct330', 'Tue1_5': 'Ct430', 'Tue1_6': 'Ct630',
        'Wed1_1': 'Cw1030', 'Wed1_2': 'Cw1230', 'Wed1_3': 'Cw230', 'Wed1_4': 'Cw430', 'Wed1_5': 'Cw630',
        'Thu1_1': 'Cth1230', 'Thu1_2': 'Cth330', 'Thu1_3': 'Cth430', 'Thu1_4': 'Cth630',

        # Second round only - Fri2_ columns
        'Fri2_1': 'Df1030', 'Fri2_2': 'Df1230', 'Fri2_3': 'Df230', 'Fri2_4': 'Df430', 'Fri2_5': 'Df630',
        'Sat2_1': 'Dsa1000', 'Sat2_2': 'Dsa100', 'Sat2_3': 'Dsa300',
        'Sun2_1': 'Dsu1000', 'Sun2_2': 'Dsu100', 'Sun2_3': 'Dsu300',
        'Mon2_1': 'Dm1030', 'Mon2_2': 'Dm1230', 'Mon2_3': 'Dm230', 'Mon2_4': 'Dm430', 'Mon2_5': 'Dm630',
        'Tue2_1': 'Dt1000', 'Tue2_2': 'Dt1100', 'Tue2_3': 'Dt1230', 'Tue2_4': 'Dt330', 'Tue2_5': 'Dt430', 'Tue2_6': 'Dt630',
        'Wed2_1': 'Dw1030', 'Wed2_2': 'Dw1230', 'Wed2_3': 'Dw230', 'Wed2_4': 'Dw430', 'Wed2_5': 'Dw630',
        # Thursday 2nd round - Q48 columns
        'Q48_1': 'Dth1230', 'Q48_2': 'Dth330', 'Q48_3': 'Dth430', 'Q48_4': 'Dth630',
    }

    # Round type constants
    ROUND_BOTH = 'B'
    ROUND_FIRST = 'C'
    ROUND_SECOND = 'D'

    # Format codes
    FORMAT_VIRTUAL_ONLY = 'Z'
    FORMAT_INPERSON_ONLY = 'P'
    FORMAT_EITHER = 'E'

    # Availability values in Qualtrics
    AVAIL_YES = 1
    AVAIL_IF_NECESSARY = 2
    AVAIL_NO = 3

    # Category mapping
    CATEGORY_MAP = {1: 'Student', 2: 'Staff', 3: 'Faculty', 4: 'Alum', 5: 'Community Member'}

    # Year in college mapping
    YEAR_MAP = {1: 'First-Year', 2: 'Sophomore', 3: 'Junior', 4: 'Senior'}

    # Gender mapping
    GENDER_MAP = {1: 'Male', 2: 'Female', 3: 'Non-binary', 4: 'Prefer not to say', 5: 'Self-describe'}

    # Ideology mapping
    IDEOLOGY_MAP = {1: 'Very Conservative', 2: 'Somewhat Conservative', 3: 'Moderate',
                    4: 'Somewhat Liberal', 5: 'Very Liberal', 6: 'Prefer not to say'}

    # Age mapping
    AGE_MAP = {1: 'Under 18', 2: '18-20', 3: '21-24', 4: '25-34', 5: '35-44',
               6: '45-54', 7: '55-64', 8: '65+', 9: 'Prefer not to say'}

    # Race mapping (individual code -> label)
    RACE_CODE_MAP = {1: 'American Indian/Alaska Native', 2: 'Asian',
                     3: 'Black/African American', 4: 'Hispanic/Latino',
                     5: 'Middle Eastern/North African', 6: 'Native Hawaiian/Pacific Islander',
                     7: 'White', 8: 'Other', 9: 'Prefer not to say', 0: 'Not specified'}

    # Source mapping (Q6)
    SOURCE_MAP = {1: 'Faculty/Staff', 2: 'Email', 3: 'Friend', 4: 'Other'}

    # Issue position mapping
    ISSUE_MAP = {1: 'Strongly Agree', 2: 'Somewhat Agree', 3: 'Neither',
                 4: 'Somewhat Disagree', 5: 'Strongly Disagree', 6: 'Prefer not to say'}

    # Format mapping
    FORMAT_MAP = {1: 'Virtual Only', 2: 'In Person Only', 3: 'Either'}

    # Rounds mapping
    ROUNDS_MAP = {1: 'Both Rounds', 2: 'First Round Only', 3: 'Second Round Only'}

    def __init__(self, registrant_data_path: str, verbose: bool = True):
        """
        Initialize the solver with registrant data.

        Args:
            registrant_data_path: Path to the Excel or CSV file with registrant data
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        self.log("=" * 70)
        self.log("D-TEAM FORMATION SOLVER v3.0 (Qualtrics Format)")
        self.log("Deliberative Citizenship Initiative - Davidson College")
        self.log("=" * 70)
        self.log("Loading registrant data...")

        # Load data - supports both CSV and Excel
        if registrant_data_path.endswith('.csv'):
            self.df_raw = pd.read_csv(registrant_data_path)
        else:
            self.df_raw = pd.read_excel(registrant_data_path)

        # The actual data starts at row 2 (0-indexed), first 2 rows are Qualtrics headers
        self.df = self.df_raw.iloc[2:].copy().reset_index(drop=True)

        # Rename the unnamed columns to meaningful names
        rename_map = {}
        for col in self.df.columns:
            if col == 'Unnamed: 0':
                rename_map[col] = 'UniqueID'
            elif col == 'Unnamed: 25':
                rename_map[col] = 'RaceBinary'
            elif col == 'Unnamed: 129':
                rename_map[col] = 'FriendInvited'
            elif col == 'Unnamed: 130':
                rename_map[col] = 'FriendInvitedBy'
            elif col == 'Unnamed: 131':
                rename_map[col] = 'Facilitator'
            elif col == 'Unnamed: 132':
                rename_map[col] = 'OnlineLastSemester'
            elif col == 'Unnamed: 133':
                rename_map[col] = 'NotPrimaryLastSemester'
        self.df.rename(columns=rename_map, inplace=True)

        # Convert numeric columns
        self._convert_numeric_columns()

        # Initialize structures
        self.team_slots = []
        self.participants = []
        self.fellows = []
        self.friend_pairs = []
        self.fellow_ids = set()

        self._extract_fellows()
        self._extract_participants()
        self._extract_friend_pairs()
        self._generate_team_slots()

        self.log(f"Loaded {len(self.participants)} eligible participants")
        self.log(f"Identified {len(self.fellows)} fellows")
        self.log(f"Found {len(self.friend_pairs)} mutual friend pair requests")
        self.log(f"Generated {len(self.team_slots)} potential team slots")

        # Log round distribution
        round_counts = {'B': 0, 'C': 0, 'D': 0}
        for p in self.participants:
            round_counts[p['round_type']] += 1
        self.log(f"  - Both rounds: {round_counts['B']} participants")
        self.log(f"  - First round only: {round_counts['C']} participants")
        self.log(f"  - Second round only: {round_counts['D']} participants")

        # Log fellow info
        returning = sum(1 for f in self.fellows if f['was_facilitator_before'])
        new_fellows = sum(1 for f in self.fellows if not f['was_facilitator_before'])
        self.log(f"  - Returning fellows: {returning}")
        self.log(f"  - New fellows: {new_fellows}")

    def log(self, message: str):
        """Print a log message if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")

    def _convert_numeric_columns(self):
        """Convert columns that should be numeric."""
        numeric_cols = ['Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Pro Liberty', 'Pro Rule of Law',
                        'Format', 'Rounds', 'UniqueID', 'FriendInvited', 'FriendInvitedBy',
                        'Facilitator', 'OnlineLastSemester', 'NotPrimaryLastSemester',
                        'RaceBinary']

        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Convert availability columns
        for col in self.df.columns:
            if any(col.startswith(prefix) for prefix in ['Fri', 'Sat', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Q48']):
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

    def _extract_fellows(self):
        """Extract DCI fellow information from the Facilitator column."""
        self.fellow_ids = set()

        for idx, row in self.df.iterrows():
            fac_val = row.get('Facilitator')
            if pd.notna(fac_val) and fac_val in [1, 2]:
                unique_id = row.get('UniqueID')
                if pd.notna(unique_id):
                    unique_id = int(unique_id)
                else:
                    unique_id = idx + 1000  # fallback

                # Facilitator column: 1 = type 1, 2 = type 2
                fellow_type_code = int(fac_val)

                # Check if they were online last semester
                online_last = row.get('OnlineLastSemester')
                was_online = (pd.notna(online_last) and online_last == 1)

                # Check if they were NOT primary last semester
                not_primary_last = row.get('NotPrimaryLastSemester')
                was_not_primary = (pd.notna(not_primary_last) and not_primary_last == 1)

                # A fellow with either flag is a "returning" fellow
                was_facilitator_before = was_online or was_not_primary

                # Get demographic data (shared with participants)
                demo = self._extract_demographic_data(row, idx)

                # Build availability for fellows
                available_slots = []
                available_if_necessary_slots = []
                round_type = demo['round_type']

                for col_name, internal_code in self.QUALTRICS_COL_MAP.items():
                    if internal_code.startswith(round_type):
                        val = row.get(col_name)
                        if pd.notna(val):
                            try:
                                val_int = int(float(val))
                                if val_int == self.AVAIL_YES:
                                    available_slots.append(internal_code)
                                elif val_int == self.AVAIL_IF_NECESSARY:
                                    available_if_necessary_slots.append(internal_code)
                            except (ValueError, TypeError):
                                pass

                # If no availability found for their round, try all round prefixes
                if not available_slots and not available_if_necessary_slots:
                    for col_name, internal_code in self.QUALTRICS_COL_MAP.items():
                        val = row.get(col_name)
                        if pd.notna(val):
                            try:
                                val_int = int(float(val))
                                if val_int == self.AVAIL_YES:
                                    available_slots.append(internal_code)
                                elif val_int == self.AVAIL_IF_NECESSARY:
                                    available_if_necessary_slots.append(internal_code)
                            except (ValueError, TypeError):
                                pass

                fellow = {
                    'id': unique_id,
                    'row_index': idx,
                    'fellow_type_code': fellow_type_code,
                    'assignment': None,
                    'available_slots': available_slots + available_if_necessary_slots,
                    'format_pref': demo['format_pref'],
                    'primary_count': 0,
                    'secondary_count': 0,
                    'was_facilitator_before': was_facilitator_before,
                    'was_online_last_semester': was_online,
                    'was_not_primary_last_semester': was_not_primary,
                    'round_type': round_type,
                    'is_fellow': True,
                    'facilitator_role': None,  # Will be set after solving

                    # All demographics
                    **demo,
                }

                self.fellows.append(fellow)
                self.fellow_ids.add(unique_id)

        # If no fellows in data, create placeholders
        if len(self.fellows) == 0:
            self.log("  WARNING: No fellows found in data. Creating placeholder fellows.")
            all_slots = list(self.QUALTRICS_COL_MAP.values())
            for i in range(40):
                self.fellows.append({
                    'id': f'fellow_{i}',
                    'row_index': -1,
                    'fellow_type_code': 1 if i % 2 == 0 else 2,
                    'assignment': None,
                    'available_slots': all_slots,
                    'format_pref': self.FORMAT_EITHER,
                    'primary_count': 0,
                    'secondary_count': 0,
                    'was_facilitator_before': False,
                    'was_online_last_semester': False,
                    'was_not_primary_last_semester': False,
                    'round_type': self.ROUND_BOTH,
                    'is_fellow': True,
                    'facilitator_role': None,
                    'is_student': False, 'is_nonwhite': False, 'is_white': True,
                    'is_male': False, 'is_female': False, 'is_nonbinary': False,
                    'is_conservative': False, 'is_liberal': False, 'is_moderate': False,
                    'ideology': '', 'ideology_code': 0, 'category': 'Fellow',
                    'issue1_agree': False, 'issue1_disagree': False,
                    'issue2_agree': False, 'issue2_disagree': False,
                    'issue1_position': '', 'issue2_position': '',
                    'format_preference_full': 'Either', 'rounds_full': 'Both Rounds',
                    'year': '', 'year_code': 0,
                    'age_range': '', 'age_code': 0, 'source': '', 'source_codes': [],
                    'race_codes': [], 'races': [], 'race_binary': 2,
                    'gender_code': 0, 'pro_liberty_code': 0, 'pro_rule_code': 0,
                })

    def _extract_demographic_data(self, row, idx) -> dict:
        """Extract demographic data from a row (shared between participants and fellows)."""
        # ========== STUDENT STATUS ==========
        cat_value = row.get('Q5')
        if pd.notna(cat_value):
            try:
                cat_value = int(float(cat_value))
            except (ValueError, TypeError):
                cat_value = None
        is_student = (cat_value == 1) if cat_value is not None else False
        category = self.CATEGORY_MAP.get(cat_value, 'Community Member') if cat_value is not None else 'Community Member'

        # ========== YEAR ==========
        year_value = row.get('Q7')
        year_code = 0
        if pd.notna(year_value):
            try:
                year_code = int(float(year_value))
            except (ValueError, TypeError):
                pass
        year = self.YEAR_MAP.get(year_code, '')

        # ========== SOURCE (Q6) ==========
        source_raw = row.get('Q6')
        source_codes = []
        source_labels = []
        if pd.notna(source_raw):
            for part in str(source_raw).split(','):
                try:
                    code = int(float(part.strip()))
                    source_codes.append(code)
                    source_labels.append(self.SOURCE_MAP.get(code, f'Code {code}'))
                except (ValueError, TypeError):
                    pass
        source = ', '.join(source_labels)

        # ========== GENDER ==========
        gender_value = row.get('Q8')
        gender_code = 0
        if pd.notna(gender_value):
            try:
                gender_code = int(float(gender_value))
            except (ValueError, TypeError):
                pass
        is_male = (gender_code == 1)
        is_female = (gender_code == 2)
        is_nonbinary = (gender_code == 3)

        # ========== AGE ==========
        age_value = row.get('Q9')
        age_code = 0
        if pd.notna(age_value):
            try:
                age_code = int(float(age_value))
            except (ValueError, TypeError):
                pass
        age_range = self.AGE_MAP.get(age_code, '')

        # ========== IDEOLOGY ==========
        ideo_value = row.get('Q10')
        ideo_code = 0
        if pd.notna(ideo_value):
            try:
                ideo_code = int(float(ideo_value))
            except (ValueError, TypeError):
                pass
        is_conservative = (ideo_code in [1, 2])
        is_liberal = (ideo_code in [4, 5])
        is_moderate = (ideo_code == 3)
        ideology = self.IDEOLOGY_MAP.get(ideo_code, '')

        # ========== RACE ==========
        race_raw = row.get('Race/Ethnicity')
        race_codes = []
        races = []
        if pd.notna(race_raw):
            for part in str(race_raw).replace(' ', '').split(','):
                try:
                    code = int(float(part))
                    race_codes.append(code)
                    races.append(self.RACE_CODE_MAP.get(code, f'Code {code}'))
                except (ValueError, TypeError):
                    pass

        race_binary_val = row.get('RaceBinary')
        race_binary = 2  # default to White
        if pd.notna(race_binary_val):
            try:
                race_binary = int(float(race_binary_val))
            except (ValueError, TypeError):
                pass

        # Determine white/non-white
        # race_binary: 0 = prefer not to respond (ignore in race constraints), 1 = non-white, 2 = white
        if race_binary == 0:
            # Prefer not to respond: don't count as either for constraint purposes
            is_white = False
            is_nonwhite = False
        elif race_binary == 1:
            is_white = False
            is_nonwhite = True
        elif race_binary == 2:
            is_white = True
            is_nonwhite = False
        else:
            # Fallback to race codes if race_binary missing
            is_white = (7 in race_codes and len(race_codes) == 1)
            is_nonwhite = bool(race_codes and 7 not in race_codes) or (len(race_codes) > 1 and 7 in race_codes)
            if not is_white and not is_nonwhite:
                is_white = False
                is_nonwhite = False

        # ========== ISSUE POSITIONS ==========
        pro_liberty = row.get('Pro Liberty')
        pro_liberty_code = 0
        if pd.notna(pro_liberty):
            try:
                pro_liberty_code = int(float(pro_liberty))
            except (ValueError, TypeError):
                pass
        issue1_agree = (pro_liberty_code in [1, 2])
        issue1_disagree = (pro_liberty_code in [4, 5])
        issue1_position = self.ISSUE_MAP.get(pro_liberty_code, '')

        pro_rule = row.get('Pro Rule of Law')
        pro_rule_code = 0
        if pd.notna(pro_rule):
            try:
                pro_rule_code = int(float(pro_rule))
            except (ValueError, TypeError):
                pass
        issue2_agree = (pro_rule_code in [1, 2])
        issue2_disagree = (pro_rule_code in [4, 5])
        issue2_position = self.ISSUE_MAP.get(pro_rule_code, '')

        # ========== FORMAT PREFERENCE ==========
        format_value = row.get('Format')
        format_code = 3  # default Either
        if pd.notna(format_value):
            try:
                format_code = int(float(format_value))
            except (ValueError, TypeError):
                pass
        if format_code == 1:
            format_pref = self.FORMAT_VIRTUAL_ONLY
        elif format_code == 2:
            format_pref = self.FORMAT_INPERSON_ONLY
        else:
            format_pref = self.FORMAT_EITHER
        format_preference_full = self.FORMAT_MAP.get(format_code, 'Either')

        # ========== ROUND PREFERENCE ==========
        rounds_value = row.get('Rounds')
        rounds_code = 1  # default Both
        if pd.notna(rounds_value):
            try:
                rounds_code = int(float(rounds_value))
            except (ValueError, TypeError):
                pass
        if rounds_code == 1:
            round_type = self.ROUND_BOTH
        elif rounds_code == 2:
            round_type = self.ROUND_FIRST
        else:
            round_type = self.ROUND_SECOND
        rounds_full = self.ROUNDS_MAP.get(rounds_code, 'Both Rounds')

        return {
            'is_student': is_student,
            'category': category,
            'year': year,
            'year_code': year_code,
            'source': source,
            'source_codes': source_codes,
            'gender_code': gender_code,
            'is_male': is_male,
            'is_female': is_female,
            'is_nonbinary': is_nonbinary,
            'age_range': age_range,
            'age_code': age_code,
            'ideology': ideology,
            'ideology_code': ideo_code,
            'is_conservative': is_conservative,
            'is_liberal': is_liberal,
            'is_moderate': is_moderate,
            'race_codes': race_codes,
            'races': races,
            'race_binary': race_binary,
            'is_white': is_white,
            'is_nonwhite': is_nonwhite,
            'pro_liberty_code': pro_liberty_code,
            'issue1_agree': issue1_agree,
            'issue1_disagree': issue1_disagree,
            'issue1_position': issue1_position,
            'pro_rule_code': pro_rule_code,
            'issue2_agree': issue2_agree,
            'issue2_disagree': issue2_disagree,
            'issue2_position': issue2_position,
            'format_pref': format_pref,
            'format_preference_full': format_preference_full,
            'round_type': round_type,
            'rounds_full': rounds_full,
        }

    def _extract_participants(self):
        """Extract participant information from the dataframe (non-fellows only)."""
        for idx, row in self.df.iterrows():
            # Skip fellows - they're handled separately
            unique_id = row.get('UniqueID')
            if pd.notna(unique_id):
                try:
                    uid = int(float(unique_id))
                except (ValueError, TypeError):
                    uid = None
                if uid is not None and uid in self.fellow_ids:
                    continue

            participant = self._create_participant_record(row, idx)

            # Only include participants with some availability
            if participant['total_available'] > 0 or participant['total_available_if_necessary'] > 0:
                self.participants.append(participant)
            else:
                self.log(f"  Skipping participant {participant['id']}: no availability")

    def _create_participant_record(self, row, idx) -> dict:
        """Create a participant record from a dataframe row."""
        unique_id = row.get('UniqueID')
        if pd.notna(unique_id):
            try:
                unique_id = int(float(unique_id))
            except (ValueError, TypeError):
                unique_id = idx + 2000
        else:
            unique_id = idx + 2000

        # Get demographic data
        demo = self._extract_demographic_data(row, idx)

        # ========== AVAILABILITY ==========
        available_slots = []
        available_if_necessary_slots = []

        slot_prefix = demo['round_type']

        for col_name, internal_code in self.QUALTRICS_COL_MAP.items():
            if internal_code.startswith(slot_prefix):
                val = row.get(col_name)
                if pd.notna(val):
                    try:
                        val_int = int(float(val))
                        if val_int == self.AVAIL_YES:
                            available_slots.append(internal_code)
                        elif val_int == self.AVAIL_IF_NECESSARY:
                            available_if_necessary_slots.append(internal_code)
                    except (ValueError, TypeError):
                        pass

        # ========== FRIEND INFO ==========
        friend_invited = None
        friend_invited_by = None

        fi_val = row.get('FriendInvited')
        if pd.notna(fi_val):
            try:
                friend_invited = int(float(fi_val))
            except (ValueError, TypeError):
                pass

        fib_val = row.get('FriendInvitedBy')
        if pd.notna(fib_val):
            try:
                friend_invited_by = int(float(fib_val))
            except (ValueError, TypeError):
                pass

        # Friend names (for display)
        friend_invited_name = ''
        for col in ['Invited_1', 'Invited_2']:
            v = row.get(col)
            if pd.notna(v) and str(v).strip():
                friend_invited_name += str(v).strip() + ' '
        friend_invited_name = friend_invited_name.strip()

        friend_invited_by_name = ''
        for col in ['Invited by_1', 'Invited by_2']:
            v = row.get(col)
            if pd.notna(v) and str(v).strip():
                friend_invited_by_name += str(v).strip() + ' '
        friend_invited_by_name = friend_invited_by_name.strip()

        # ========== BUILD PARTICIPANT RECORD ==========
        participant = {
            'id': unique_id,
            'row_index': idx,
            'is_fellow': False,
            'facilitator_role': None,

            # All demographics
            **demo,

            # Availability
            'available_slots': available_slots,
            'available_if_necessary_slots': available_if_necessary_slots,
            'all_possible_slots': available_slots + available_if_necessary_slots,
            'total_available': len(available_slots),
            'total_available_if_necessary': len(available_if_necessary_slots),

            # Friend info
            'friend_invited': friend_invited,
            'friend_invited_by': friend_invited_by,
            'friend_invited_name': friend_invited_name,
            'friend_invited_by_name': friend_invited_by_name,

            # Compatibility fields
            'taking_for_credit': False,
            'courses': [],
            'connection': demo['source'],
            'date_submitted': '',
            'status': 'Registered',
        }

        return participant

    def _extract_friend_pairs(self):
        """
        Extract friend pair requests with MUTUAL CONSENT requirement.
        Both friends must list each other for the pairing to be valid.
        A invites B AND B says invited-by A => mutual pair.
        """
        participant_ids = {p['id'] for p in self.participants}

        # Build maps
        invited_map = {}    # pid -> set of friend IDs they invited
        invited_by_map = {} # pid -> set of friend IDs who invited them

        for p in self.participants:
            pid = p['id']
            if p['friend_invited'] is not None and p['friend_invited'] in participant_ids:
                invited_map.setdefault(pid, set()).add(p['friend_invited'])
            if p['friend_invited_by'] is not None and p['friend_invited_by'] in participant_ids:
                invited_by_map.setdefault(pid, set()).add(p['friend_invited_by'])

        # Find mutual pairs
        mutual_pairs = set()
        for pid, friends in invited_map.items():
            for friend_id in friends:
                # Check: did the friend say they were invited by this person?
                if friend_id in invited_by_map and pid in invited_by_map[friend_id]:
                    pair = tuple(sorted([pid, friend_id]))
                    mutual_pairs.add(pair)
                # OR: did the friend also invite this person?
                if friend_id in invited_map and pid in invited_map[friend_id]:
                    pair = tuple(sorted([pid, friend_id]))
                    mutual_pairs.add(pair)

        self.friend_pairs = list(mutual_pairs)

        # Log non-mutual requests
        all_requests = set()
        for pid, friends in invited_map.items():
            for fid in friends:
                all_requests.add(tuple(sorted([pid, fid])))
        for pid, friends in invited_by_map.items():
            for fid in friends:
                all_requests.add(tuple(sorted([pid, fid])))

        non_mutual = all_requests - mutual_pairs
        if non_mutual:
            self.log(f"  {len(non_mutual)} non-mutual friend requests ignored (consent required from both)")

    def _generate_team_slots(self):
        """Generate all possible team slots (time + format + round combinations)."""
        round_slots = {'B': set(), 'C': set(), 'D': set()}

        for col_name, internal_code in self.QUALTRICS_COL_MAP.items():
            round_type = internal_code[0]
            round_slots[round_type].add(internal_code)

        for round_type, slots in round_slots.items():
            round_name = {'B': 'Both Rounds', 'C': 'First Round', 'D': 'Second Round'}[round_type]

            for slot_code in sorted(slots):
                time_code = slot_code[1:]

                if time_code in self.TIME_SLOT_MAP_QUALTRICS:
                    day, time = self.TIME_SLOT_MAP_QUALTRICS[time_code]
                else:
                    day = self._infer_day(time_code)
                    time = self._infer_time(time_code)

                # In-Person team
                self.team_slots.append({
                    'id': f'{slot_code}p',
                    'slot_code': slot_code,
                    'day': day,
                    'time': time,
                    'is_virtual': False,
                    'round_type': round_type,
                    'display_name': f'{day} {time} ({round_name}, In-Person)',
                })

                # Virtual team
                self.team_slots.append({
                    'id': f'{slot_code}v',
                    'slot_code': slot_code,
                    'day': day,
                    'time': time,
                    'is_virtual': True,
                    'round_type': round_type,
                    'display_name': f'{day} {time} ({round_name}, Virtual)',
                })

    def _infer_day(self, time_code: str) -> str:
        """Infer day of week from time code prefix."""
        if time_code.startswith('f'):
            return 'Friday'
        elif time_code.startswith('sa'):
            return 'Saturday'
        elif time_code.startswith('su'):
            return 'Sunday'
        elif time_code.startswith('m'):
            return 'Monday'
        elif time_code.startswith('th'):
            return 'Thursday'
        elif time_code.startswith('t'):
            return 'Tuesday'
        elif time_code.startswith('w'):
            return 'Wednesday'
        return 'Unknown'

    def _infer_time(self, time_code: str) -> str:
        """Infer time range from time code numbers."""
        numbers = ''.join(c for c in time_code if c.isdigit())
        if not numbers:
            return 'Unknown'
        hour = int(numbers[:len(numbers)//2 + len(numbers) % 2]) if len(numbers) >= 2 else int(numbers)
        if hour < 12:
            return f'{hour}:00 AM - {hour+2}:00 AM'
        elif hour == 12:
            return '12:00 PM - 2:00 PM'
        else:
            hour12 = hour if hour <= 12 else hour - 12
            return f'{hour12}:00 PM - {hour12+2}:00 PM'

    def _can_participant_join_team(self, participant: dict, team: dict,
                                   use_if_necessary: bool = True) -> Tuple[bool, bool]:
        """
        Check if a participant can join a given team slot.
        Returns (can_join, is_preferred) where is_preferred means "Available" not just "If Necessary".
        
        'Both Rounds' (B) participants can also join First-Only (C) and 
        Second-Only (D) team slots, since they're available for both rounds.
        """
        # Round type must match (or participant is "Both" which covers C and D)
        p_round = participant['round_type']
        t_round = team['round_type']
        if p_round != t_round and p_round != self.ROUND_BOTH:
            return False, False

        slot_code = team['slot_code']
        
        # For "Both" participants joining C/D teams, check the B-version of the slot
        if p_round == self.ROUND_BOTH and t_round != self.ROUND_BOTH:
            time_part = slot_code[1:]  # e.g., 'm630' from 'Cm630'
            check_code = 'B' + time_part  # e.g., 'Bm630'
        else:
            check_code = slot_code
        
        is_available = check_code in participant['available_slots']
        is_available_if_necessary = check_code in participant.get('available_if_necessary_slots', [])

        if not is_available and not (use_if_necessary and is_available_if_necessary):
            return False, False

        # Format compatibility
        format_pref = participant['format_pref']
        if format_pref == self.FORMAT_VIRTUAL_ONLY and not team['is_virtual']:
            return False, False
        if format_pref == self.FORMAT_INPERSON_ONLY and team['is_virtual']:
            return False, False

        return True, is_available

    def _can_fellow_facilitate_team(self, fellow: dict, team: dict) -> bool:
        """Check if a fellow can facilitate a given team slot.
        
        'Both Rounds' (B) fellows can also facilitate First-Only (C) and 
        Second-Only (D) team slots, since they're available for both rounds.
        We check if the fellow is available for the same time slot (ignoring 
        round prefix) to enable cross-round staffing.
        """
        slot_code = team['slot_code']  # e.g., 'Cm630' or 'Dm630'
        
        # Direct match: fellow has this exact slot
        if slot_code in fellow['available_slots']:
            pass  # OK, matched
        elif fellow['round_type'] == self.ROUND_BOTH:
            # "Both Rounds" fellow → check if they have the B-version of this time slot
            time_part = slot_code[1:]  # e.g., 'm630' from 'Cm630'
            b_slot = 'B' + time_part   # e.g., 'Bm630'
            if b_slot not in fellow['available_slots']:
                return False
        else:
            return False

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
            min_team_size: Minimum desired participants per team (not counting fellows)
            max_team_size: Maximum desired participants per team (not counting fellows)
            allow_flexible_size: Allow min-1 and max+1 as last resort
            time_limit_seconds: Maximum solver runtime
            prioritize_low_availability: Give bonus to participants with fewer available slots

        Returns:
            Solution dictionary with teams, assignments, and statistics
        """
        self.log("")
        self.log("=" * 70)
        self.log("STARTING OPTIMIZATION")
        self.log("=" * 70)

        actual_min = max(min_team_size - 3, 5) if allow_flexible_size else min_team_size
        actual_max = max_team_size + 2 if allow_flexible_size else max_team_size

        self.log(f"Team size target: {min_team_size}-{max_team_size} (flexible: {actual_min}-{actual_max})")

        # Create the optimization problem (maximization)
        prob = LpProblem("DTeam_Formation_V3", LpMaximize)

        # ================================================================
        # DECISION VARIABLES
        # ================================================================
        self.log("Creating decision variables...")

        # x[p_id][t_id] = 1 if participant p is assigned to team t
        x = {}
        preferred = {}

        for p in self.participants:
            x[p['id']] = {}
            preferred[p['id']] = {}
            for t in self.team_slots:
                can_join, is_preferred = self._can_participant_join_team(p, t)
                if can_join:
                    x[p['id']][t['id']] = LpVariable(f"x_{p['id']}_{t['id']}", cat='Binary')
                    preferred[p['id']][t['id']] = is_preferred

        # y[t_id] = 1 if team t is formed
        y = {}
        for t in self.team_slots:
            y[t['id']] = LpVariable(f"y_{t['id']}", cat='Binary')

        # f_primary[f_id][t_id] = 1 if fellow f is Primary facilitator for team t
        # f_secondary[f_id][t_id] = 1 if fellow f is Secondary facilitator for team t
        f_primary = {}
        f_secondary = {}
        for f in self.fellows:
            f_primary[f['id']] = {}
            f_secondary[f['id']] = {}
            for t in self.team_slots:
                if self._can_fellow_facilitate_team(f, t):
                    f_primary[f['id']][t['id']] = LpVariable(f"fp_{f['id']}_{t['id']}", cat='Binary')
                    f_secondary[f['id']][t['id']] = LpVariable(f"fs_{f['id']}_{t['id']}", cat='Binary')

        # Slack variables for soft constraints
        s_size_under = {t['id']: LpVariable(f"s_under_{t['id']}", lowBound=0, cat='Integer') for t in self.team_slots}
        s_size_over = {t['id']: LpVariable(f"s_over_{t['id']}", lowBound=0, cat='Integer') for t in self.team_slots}
        s_women = {t['id']: LpVariable(f"s_women_{t['id']}", lowBound=0, cat='Integer') for t in self.team_slots}
        s_men = {t['id']: LpVariable(f"s_men_{t['id']}", lowBound=0, cat='Integer') for t in self.team_slots}
        s_conservative = {t['id']: LpVariable(f"s_cons_{t['id']}", lowBound=0, cat='Integer') for t in self.team_slots}
        s_liberal = {t['id']: LpVariable(f"s_lib_{t['id']}", lowBound=0, cat='Integer') for t in self.team_slots}
        s_white = {t['id']: LpVariable(f"s_white_{t['id']}", lowBound=0, cat='Integer') for t in self.team_slots}
        s_nonwhite = {t['id']: LpVariable(f"s_nw_{t['id']}", lowBound=0, cat='Integer') for t in self.team_slots}
        s_students = {t['id']: LpVariable(f"s_stu_{t['id']}", lowBound=0, cat='Integer') for t in self.team_slots}
        s_nonstudents = {t['id']: LpVariable(f"s_nstu_{t['id']}", lowBound=0, cat='Integer') for t in self.team_slots}
        s_issue1_agree = {t['id']: LpVariable(f"s_i1a_{t['id']}", lowBound=0, cat='Integer') for t in self.team_slots}
        s_issue1_disagree = {t['id']: LpVariable(f"s_i1d_{t['id']}", lowBound=0, cat='Integer') for t in self.team_slots}
        s_issue2_agree = {t['id']: LpVariable(f"s_i2a_{t['id']}", lowBound=0, cat='Integer') for t in self.team_slots}
        s_issue2_disagree = {t['id']: LpVariable(f"s_i2d_{t['id']}", lowBound=0, cat='Integer') for t in self.team_slots}

        # Friend pair bonus variables
        friend_bonus = {}
        for i, (p1, p2) in enumerate(self.friend_pairs):
            friend_bonus[i] = {}
            for t in self.team_slots:
                friend_bonus[i][t['id']] = LpVariable(f"friend_{i}_{t['id']}", cat='Binary')

        # ================================================================
        # OBJECTIVE FUNCTION
        # ================================================================
        self.log("Setting up objective function...")

        # Weights (higher = more important)
        W_PARTICIPANT = 1000       # Base reward for assigning a participant
        W_HARD_STUDENT = 500       # Penalty for violating student minimum (soft - fellows count too)
        W_HARD_NONSTU = 500        # Penalty for violating non-student minimum (soft - fellows count too)
        W_SIZE_UNDER = 350         # Penalty per person under TARGET size (per person: 1 short of 8 = 350 penalty)
        W_SIZE_OVER = 60           # Penalty for being over max team size
        W_WOMEN = 60               # Penalty for not having 2 women
        W_MEN = 40                 # Penalty for not having 2 men
        W_CONSERVATIVE = 50        # Penalty for no conservative
        W_LIBERAL = 50             # Penalty for no liberal
        W_WHITE = 40               # Penalty for no white participant
        W_NONWHITE = 50            # Penalty for no non-white participant
        W_ISSUE1_AGREE = 30        # Penalty for no Pro Liberty agree
        W_ISSUE1_DISAGREE = 30     # Penalty for no Pro Liberty disagree
        W_ISSUE2_AGREE = 30        # Penalty for no Pro Rule agree
        W_ISSUE2_DISAGREE = 30     # Penalty for no Pro Rule disagree
        W_FRIEND = 200             # Bonus for placing friend pairs together
        W_INPERSON_PREF = 10       # Small bonus for putting either-format in person
        W_LOW_AVAIL_BONUS = 50     # Bonus for participants with few available slots
        W_PREFERRED_SLOT = 5       # Small bonus for "Available" over "If Necessary"
        W_NEW_FELLOW_BONUS = 30    # Bonus for assigning new (non-returning) fellows
        W_ONLINE_FELLOW_INPERSON = 25  # Bonus for online-last-semester fellows going in-person

        objective = 0

        # Reward for each participant assigned
        for p in self.participants:
            for t_id in x[p['id']]:
                base_value = W_PARTICIPANT

                # Low availability bonus
                total_slots = p['total_available'] + p['total_available_if_necessary']
                if prioritize_low_availability and total_slots <= 5:
                    base_value += W_LOW_AVAIL_BONUS * (6 - total_slots)

                objective += base_value * x[p['id']][t_id]

                # Preferred slot bonus
                if preferred[p['id']][t_id]:
                    objective += W_PREFERRED_SLOT * x[p['id']][t_id]

                # In-person preference for "Either" format
                team = next(t for t in self.team_slots if t['id'] == t_id)
                if p['format_pref'] == self.FORMAT_EITHER and not team['is_virtual']:
                    objective += W_INPERSON_PREF * x[p['id']][t_id]

        # Penalties for soft constraint violations
        for t in self.team_slots:
            objective -= W_HARD_STUDENT * s_students[t['id']]
            objective -= W_HARD_NONSTU * s_nonstudents[t['id']]
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

        # Friend pair bonuses
        for i in friend_bonus:
            for t_id in friend_bonus[i]:
                objective += W_FRIEND * friend_bonus[i][t_id]

        # New fellow priority bonus
        for f in self.fellows:
            for t_id in f_primary.get(f['id'], {}):
                if not f['was_facilitator_before']:
                    objective += W_NEW_FELLOW_BONUS * f_primary[f['id']][t_id]
            for t_id in f_secondary.get(f['id'], {}):
                if not f['was_facilitator_before']:
                    objective += W_NEW_FELLOW_BONUS * f_secondary[f['id']][t_id]

        # Online-last-semester fellows should be prioritized for in-person teams
        for f in self.fellows:
            if f.get('was_online_last_semester', False):
                for t_id in f_primary.get(f['id'], {}):
                    team = next(t for t in self.team_slots if t['id'] == t_id)
                    if not team['is_virtual']:
                        objective += W_ONLINE_FELLOW_INPERSON * f_primary[f['id']][t_id]
                for t_id in f_secondary.get(f['id'], {}):
                    team = next(t for t in self.team_slots if t['id'] == t_id)
                    if not team['is_virtual']:
                        objective += W_ONLINE_FELLOW_INPERSON * f_secondary[f['id']][t_id]

        prob += objective, "Total_Objective"

        # ================================================================
        # CONSTRAINTS
        # ================================================================
        self.log("Adding constraints...")

        # ---- HARD CONSTRAINT 1: Each participant in at most 1 team ----
        for p in self.participants:
            prob += (
                lpSum(x[p['id']][t_id] for t_id in x[p['id']]) <= 1,
                f"HC1_one_team_{p['id']}"
            )

        # ---- HARD CONSTRAINT 2: Team formation linkage ----
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

        # ---- HARD CONSTRAINT 3: Each team needs 1 Primary and 1 Secondary ----
        for t in self.team_slots:
            eligible_p = [f for f in self.fellows if t['id'] in f_primary.get(f['id'], {})]
            eligible_s = [f for f in self.fellows if t['id'] in f_secondary.get(f['id'], {})]
            if eligible_p:
                prob += (
                    lpSum(f_primary[f['id']][t['id']] for f in eligible_p) == y[t['id']],
                    f"HC3_primary_{t['id']}"
                )
            else:
                # No fellows can be primary here → team CANNOT form
                prob += (y[t['id']] == 0, f"HC3_no_primary_{t['id']}")
            if eligible_s:
                prob += (
                    lpSum(f_secondary[f['id']][t['id']] for f in eligible_s) == y[t['id']],
                    f"HC3_secondary_{t['id']}"
                )
            else:
                # No fellows can be secondary here → team CANNOT form
                prob += (y[t['id']] == 0, f"HC3_no_secondary_{t['id']}")

        # ---- HARD CONSTRAINT 3b: Limit total teams to what fellow pool can sustain ----
        # With N fellows, each doing primary≤2 and secondary≤2, practical max is about N
        # But also limit by participant count: we want teams of at least min_team_size
        max_by_fellows = len(self.fellows)
        max_by_participants = len(self.participants) // actual_min + 1
        max_possible_teams = min(max_by_fellows, max_by_participants)
        prob += (
            lpSum(y[t['id']] for t in self.team_slots) <= max_possible_teams,
            "HC3b_max_teams"
        )
        self.log(f"  Max teams limited to {max_possible_teams} (min of {max_by_fellows} fellows, {max_by_participants} by participants)")

        # ---- HARD CONSTRAINT 4: No fellow as both Primary and Secondary for same team ----
        for f in self.fellows:
            for t in self.team_slots:
                if t['id'] in f_primary.get(f['id'], {}) and t['id'] in f_secondary.get(f['id'], {}):
                    prob += (
                        f_primary[f['id']][t['id']] + f_secondary[f['id']][t['id']] <= 1,
                        f"HC4_role_{f['id']}_{t['id']}"
                    )

        # ---- HARD CONSTRAINT 5: Limit assignments per fellow ----
        max_primary = 3
        max_secondary = 3
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

        # ---- HARD CONSTRAINT 5b: Fellows who were NOT Primary last semester MUST be Primary ≥1 ----
        for f in self.fellows:
            if f.get('was_not_primary_last_semester', False):
                primary_teams = [t_id for t_id in f_primary.get(f['id'], {})]
                if primary_teams:
                    prob += (
                        lpSum(f_primary[f['id']][t_id] for t_id in primary_teams) >= 1,
                        f"HC5b_must_primary_{f['id']}"
                    )
                    self.log(f"  HARD: Fellow {f['id']} must be Primary (was not Primary last semester)")

        # ---- HARD CONSTRAINT 6: At least 2 students per team ----
        for t in self.team_slots:
            students = [p for p in self.participants if t['id'] in x[p['id']] and p['is_student']]
            if students:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in students) + s_students[t['id']] >= 2 * y[t['id']],
                    f"HC6_students_{t['id']}"
                )
            else:
                prob += (y[t['id']] == 0, f"HC6_no_students_{t['id']}")

        # ---- HARD CONSTRAINT 7: At least 2 non-students per team ----
        for t in self.team_slots:
            nonstudents = [p for p in self.participants if t['id'] in x[p['id']] and not p['is_student']]
            if nonstudents:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in nonstudents) + s_nonstudents[t['id']] >= 2 * y[t['id']],
                    f"HC7_nonstudents_{t['id']}"
                )
            else:
                prob += (y[t['id']] == 0, f"HC7_no_nonstudents_{t['id']}")

        # ---- SOFT CONSTRAINTS ----

        # SC1: Team size within range (penalize shortfall from TARGET, not floor)
        for t in self.team_slots:
            eligible = [p for p in self.participants if t['id'] in x[p['id']]]
            if eligible:
                # Penalize shortfall from TARGET min_team_size (e.g., 8)
                prob += (
                    lpSum(x[p['id']][t['id']] for p in eligible) + s_size_under[t['id']] >= min_team_size * y[t['id']],
                    f"SC1_min_{t['id']}"
                )
                prob += (
                    lpSum(x[p['id']][t['id']] for p in eligible) - s_size_over[t['id']] <= actual_max * y[t['id']],
                    f"SC1_max_{t['id']}"
                )
                # Hard floor: if team forms, must have at least actual_min participants
                prob += (
                    lpSum(x[p['id']][t['id']] for p in eligible) >= actual_min * y[t['id']],
                    f"HC_floor_{t['id']}"
                )

        # SC2: At least 2 women per team
        for t in self.team_slots:
            women = [p for p in self.participants if t['id'] in x[p['id']] and p['is_female']]
            if women:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in women) + s_women[t['id']] >= 2 * y[t['id']],
                    f"SC2_women_{t['id']}"
                )

        # SC3: At least 2 men per team
        for t in self.team_slots:
            men = [p for p in self.participants if t['id'] in x[p['id']] and p['is_male']]
            if men:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in men) + s_men[t['id']] >= 2 * y[t['id']],
                    f"SC3_men_{t['id']}"
                )

        # SC4: At least 1 conservative
        for t in self.team_slots:
            conservatives = [p for p in self.participants if t['id'] in x[p['id']] and p['is_conservative']]
            if conservatives:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in conservatives) + s_conservative[t['id']] >= y[t['id']],
                    f"SC4_cons_{t['id']}"
                )
            else:
                prob += (s_conservative[t['id']] >= y[t['id']], f"SC4_no_cons_{t['id']}")

        # SC5: At least 1 liberal
        for t in self.team_slots:
            liberals = [p for p in self.participants if t['id'] in x[p['id']] and p['is_liberal']]
            if liberals:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in liberals) + s_liberal[t['id']] >= y[t['id']],
                    f"SC5_lib_{t['id']}"
                )
            else:
                prob += (s_liberal[t['id']] >= y[t['id']], f"SC5_no_lib_{t['id']}")

        # SC6: At least 1 white participant
        for t in self.team_slots:
            whites = [p for p in self.participants if t['id'] in x[p['id']] and p['is_white']]
            if whites:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in whites) + s_white[t['id']] >= y[t['id']],
                    f"SC6_white_{t['id']}"
                )
            else:
                prob += (s_white[t['id']] >= y[t['id']], f"SC6_no_white_{t['id']}")

        # SC7: At least 1 non-white participant
        for t in self.team_slots:
            nonwhites = [p for p in self.participants if t['id'] in x[p['id']] and p['is_nonwhite']]
            if nonwhites:
                prob += (
                    lpSum(x[p['id']][t['id']] for p in nonwhites) + s_nonwhite[t['id']] >= y[t['id']],
                    f"SC7_nw_{t['id']}"
                )
            else:
                prob += (s_nonwhite[t['id']] >= y[t['id']], f"SC7_no_nw_{t['id']}")

        # SC8-11: Issue positions
        for t in self.team_slots:
            i1a = [p for p in self.participants if t['id'] in x[p['id']] and p['issue1_agree']]
            i1d = [p for p in self.participants if t['id'] in x[p['id']] and p['issue1_disagree']]
            i2a = [p for p in self.participants if t['id'] in x[p['id']] and p['issue2_agree']]
            i2d = [p for p in self.participants if t['id'] in x[p['id']] and p['issue2_disagree']]
            for plist, slack, name in [
                (i1a, s_issue1_agree, "i1a"),
                (i1d, s_issue1_disagree, "i1d"),
                (i2a, s_issue2_agree, "i2a"),
                (i2d, s_issue2_disagree, "i2d"),
            ]:
                if plist:
                    prob += (
                        lpSum(x[p['id']][t['id']] for p in plist) + slack[t['id']] >= y[t['id']],
                        f"SC_{name}_{t['id']}"
                    )
                else:
                    prob += (slack[t['id']] >= y[t['id']], f"SC_{name}_none_{t['id']}")

        # SC12: Friend pair placement
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

        self.log(f"Solver status: {LpStatus[prob.status]}")
        self.log(f"Objective value: {value(prob.objective)}")

        # Extract solution
        solution = self._extract_solution(prob, x, y, f_primary, f_secondary, friend_bonus, preferred)

        return solution

    def _extract_solution(self, prob, x, y, f_primary, f_secondary, friend_bonus, preferred) -> dict:
        """Extract and organize the solution from the solved problem."""
        solution = {
            'status': LpStatus[prob.status],
            'objective_value': value(prob.objective),
            'teams': {},
            'unassigned_participants': [],
            'friend_pairs_satisfied': [],
            'friend_pairs_unsatisfied': [],
            'statistics': {},
            'constraint_violations': {'hard': [], 'soft': []},
            'facilitator_assignments': [],
            'fellows_data': [],  # All fellows with their data for the frontend
        }

        assigned_to_preferred = 0
        assigned_to_if_necessary = 0

        # Track fellow assignments
        fellow_assignment_map = {}  # fellow_id -> list of {team_id, role}

        # Extract teams
        for t in self.team_slots:
            if y[t['id']].varValue and y[t['id']].varValue > 0.5:
                members = []
                for p in self.participants:
                    if t['id'] in x[p['id']] and x[p['id']][t['id']].varValue and x[p['id']][t['id']].varValue > 0.5:
                        if preferred[p['id']][t['id']]:
                            assigned_to_preferred += 1
                            p['assigned_to_preferred'] = True
                        else:
                            assigned_to_if_necessary += 1
                            p['assigned_to_preferred'] = False
                        members.append(p)

                if members:
                    primary = None
                    secondary = None
                    for f in self.fellows:
                        if t['id'] in f_primary.get(f['id'], {}) and f_primary[f['id']][t['id']].varValue and f_primary[f['id']][t['id']].varValue > 0.5:
                            primary = f
                            f['facilitator_role'] = 'Primary'
                            fellow_assignment_map.setdefault(f['id'], []).append({'team_id': t['id'], 'role': 'Primary'})
                        if t['id'] in f_secondary.get(f['id'], {}) and f_secondary[f['id']][t['id']].varValue and f_secondary[f['id']][t['id']].varValue > 0.5:
                            secondary = f
                            f['facilitator_role'] = 'Secondary'
                            fellow_assignment_map.setdefault(f['id'], []).append({'team_id': t['id'], 'role': 'Secondary'})

                    # Include fellows in composition analysis since they are on the team
                    all_team_members = list(members)
                    if primary:
                        all_team_members.append(primary)
                    if secondary:
                        all_team_members.append(secondary)

                    solution['teams'][t['id']] = {
                        'info': t,
                        'members': members,
                        'size': len(members),
                        'primary_facilitator': primary,
                        'secondary_facilitator': secondary,
                        'composition': self._analyze_composition(all_team_members),
                    }

        # Build facilitator assignments list with full data
        for f in self.fellows:
            assignments = fellow_assignment_map.get(f['id'], [])
            primary_assignments = [a for a in assignments if a['role'] == 'Primary']
            secondary_assignments = [a for a in assignments if a['role'] == 'Secondary']

            fa_entry = {
                'fellow': f,
                'primary_teams': [a['team_id'] for a in primary_assignments],
                'secondary_teams': [a['team_id'] for a in secondary_assignments],
                'total_assignments': len(assignments),
                'was_facilitator_before': f['was_facilitator_before'],
                'was_online_last_semester': f.get('was_online_last_semester', False),
                'was_not_primary_last_semester': f.get('was_not_primary_last_semester', False),
            }
            solution['facilitator_assignments'].append(fa_entry)

        # Build fellows_data for frontend display
        for f in self.fellows:
            assignments = fellow_assignment_map.get(f['id'], [])
            roles_assigned = [a['role'] for a in assignments]
            solution['fellows_data'].append({
                'id': f['id'],
                'is_fellow': True,
                'fellow_type_code': f.get('fellow_type_code', 0),
                'facilitator_role': ', '.join(roles_assigned) if roles_assigned else 'Unassigned',
                'was_facilitator_before': f['was_facilitator_before'],
                'was_online_last_semester': f.get('was_online_last_semester', False),
                'was_not_primary_last_semester': f.get('was_not_primary_last_semester', False),
                'category': f.get('category', 'Fellow'),
                'year': f.get('year', ''),
                'age_range': f.get('age_range', ''),
                'ideology': f.get('ideology', ''),
                'gender_code': f.get('gender_code', 0),
                'is_male': f.get('is_male', False),
                'is_female': f.get('is_female', False),
                'is_student': f.get('is_student', False),
                'races': f.get('races', []),
                'issue1_position': f.get('issue1_position', ''),
                'issue2_position': f.get('issue2_position', ''),
                'source': f.get('source', ''),
                'format_preference_full': f.get('format_preference_full', ''),
                'rounds_full': f.get('rounds_full', ''),
                'total_assignments': len(assignments),
                'primary_teams': [a['team_id'] for a in assignments if a['role'] == 'Primary'],
                'secondary_teams': [a['team_id'] for a in assignments if a['role'] == 'Secondary'],
            })

        # Unassigned participants
        assigned_ids = set()
        for team in solution['teams'].values():
            for m in team['members']:
                assigned_ids.add(m['id'])

        for p in self.participants:
            if p['id'] not in assigned_ids:
                solution['unassigned_participants'].append(p)

        # Friend pairs
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
            'assigned_to_preferred_slot': assigned_to_preferred,
            'assigned_to_if_necessary_slot': assigned_to_if_necessary,
            'total_fellows': len(self.fellows),
            'fellows_assigned': len([f for f in solution['facilitator_assignments'] if f['total_assignments'] > 0]),
            'new_fellows_assigned': len([f for f in solution['facilitator_assignments'] if f['total_assignments'] > 0 and not f['was_facilitator_before']]),
            'returning_fellows_assigned': len([f for f in solution['facilitator_assignments'] if f['total_assignments'] > 0 and f['was_facilitator_before']]),
        }

        # Check violations
        solution['constraint_violations'] = self._check_violations(solution)

        return solution

    def _analyze_composition(self, members: List[dict]) -> dict:
        """Analyze the demographic composition of a team's members."""
        return {
            'students': sum(1 for m in members if m['is_student']),
            'non_students': sum(1 for m in members if not m['is_student']),
            'women': sum(1 for m in members if m['is_female']),
            'men': sum(1 for m in members if m['is_male']),
            'nonbinary': sum(1 for m in members if m.get('is_nonbinary', False)),
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
        """Check for constraint violations in the solution."""
        violations = {'hard': [], 'soft': []}
        for tid, team in solution['teams'].items():
            c = team['composition']
            s = team['size']

            # Hard violations
            if c['students'] < 2:
                violations['hard'].append(f"{tid}: {c['students']} students (need >= 2)")
            if c['non_students'] < 2:
                violations['hard'].append(f"{tid}: {c['non_students']} non-students (need >= 2)")
            if team['primary_facilitator'] is None:
                violations['hard'].append(f"{tid}: No primary facilitator")
            if team['secondary_facilitator'] is None:
                violations['hard'].append(f"{tid}: No secondary facilitator")

            # Soft violations
            if s < 7:
                violations['soft'].append(f"{tid}: Size {s} < 7")
            elif s < 8:
                violations['soft'].append(f"{tid}: Size {s} < 8 (acceptable)")
            if s > 11:
                violations['soft'].append(f"{tid}: Size {s} > 11")
            elif s > 10:
                violations['soft'].append(f"{tid}: Size {s} > 10 (acceptable)")
            if c['women'] < 2:
                violations['soft'].append(f"{tid}: {c['women']} women (want >= 2)")
            if c['men'] < 2:
                violations['soft'].append(f"{tid}: {c['men']} men (want >= 2)")
            if c['conservatives'] < 1:
                violations['soft'].append(f"{tid}: No conservatives")
            if c['liberals'] < 1:
                violations['soft'].append(f"{tid}: No liberals")
            if c['white'] < 1:
                violations['soft'].append(f"{tid}: No white participants")
            if c['non_white'] < 1:
                violations['soft'].append(f"{tid}: No non-white participants")
            if c['issue1_agree'] < 1:
                violations['soft'].append(f"{tid}: No one agreeing on Pro Liberty")
            if c['issue1_disagree'] < 1:
                violations['soft'].append(f"{tid}: No one disagreeing on Pro Liberty")
            if c['issue2_agree'] < 1:
                violations['soft'].append(f"{tid}: No one agreeing on Pro Rule of Law")
            if c['issue2_disagree'] < 1:
                violations['soft'].append(f"{tid}: No one disagreeing on Pro Rule of Law")

        return violations

    def print_report(self, solution: dict):
        """Print a comprehensive solution report."""
        print("\n" + "=" * 80)
        print("D-TEAM FORMATION SOLUTION REPORT (v3 - Qualtrics Format)")
        print("=" * 80)

        stats = solution['statistics']
        print(f"\n{'SUMMARY':^80}")
        print("-" * 80)
        print(f"  Status: {solution['status']}")
        print(f"  Participants: {stats['assigned']}/{stats['total_participants']} assigned ({stats['assignment_rate']:.1f}%)")
        print(f"  Teams Formed: {stats['teams_formed']}")
        print(f"  Friend Pairs: {stats['friend_pairs_satisfied']}/{stats['friend_pairs_total']} satisfied")
        print(f"  Assigned to 'Available' slots: {stats['assigned_to_preferred_slot']}")
        print(f"  Assigned to 'Available if Necessary' slots: {stats['assigned_to_if_necessary_slot']}")
        print(f"  Fellows: {stats['fellows_assigned']}/{stats['total_fellows']} assigned")
        print(f"    New fellows assigned: {stats['new_fellows_assigned']}")
        print(f"    Returning fellows assigned: {stats['returning_fellows_assigned']}")

        # Teams by round
        round_teams = {'B': [], 'C': [], 'D': []}
        for tid, team in solution['teams'].items():
            round_teams[team['info']['round_type']].append(tid)
        print(f"\n  Teams by Round:")
        print(f"    Both Rounds: {len(round_teams['B'])} teams")
        print(f"    First Round Only: {len(round_teams['C'])} teams")
        print(f"    Second Round Only: {len(round_teams['D'])} teams")

        # Individual teams
        print(f"\n{'TEAMS':^80}")
        print("-" * 80)
        for tid in sorted(solution['teams'].keys()):
            team = solution['teams'][tid]
            t = team['info']
            c = team['composition']

            status_icons = []
            if c['students'] < 2 or c['non_students'] < 2:
                status_icons.append("[HARD]")
            elif c['women'] < 2 or c['conservatives'] < 1 or c['liberals'] < 1 or c['non_white'] < 1:
                status_icons.append("[SOFT]")
            else:
                status_icons.append("[OK]")
            status = " ".join(status_icons)
            round_name = {'B': 'Both', 'C': '1st', 'D': '2nd'}[t['round_type']]

            print(f"\n  {status} Team: {tid}")
            print(f"     Schedule: {t['day']} {t['time']} ({'Virtual' if t['is_virtual'] else 'In-Person'}) - Round: {round_name}")
            print(f"     Size: {team['size']}")

            pf = team['primary_facilitator']
            sf = team['secondary_facilitator']
            pf_label = f"{pf['id']}" + (' (New)' if pf and not pf['was_facilitator_before'] else ' (Returning)') if pf else 'NONE'
            sf_label = f"{sf['id']}" + (' (New)' if sf and not sf['was_facilitator_before'] else ' (Returning)') if sf else 'NONE'
            print(f"     Facilitators: Primary={pf_label}, Secondary={sf_label}")

            print(f"     Composition:")
            print(f"       Students/Non-Students: {c['students']}/{c['non_students']}")
            print(f"       Women/Men/Non-binary: {c['women']}/{c['men']}/{c.get('nonbinary', 0)}")
            print(f"       Conservative/Liberal/Moderate: {c['conservatives']}/{c['liberals']}/{c['moderates']}")
            print(f"       White/Non-White: {c['white']}/{c['non_white']}")
            print(f"       Pro Liberty Agree/Disagree: {c['issue1_agree']}/{c['issue1_disagree']}")
            print(f"       Pro Rule of Law Agree/Disagree: {c['issue2_agree']}/{c['issue2_disagree']}")

        # Facilitator Stats
        print(f"\n{'FACILITATOR ASSIGNMENTS':^80}")
        print("-" * 80)
        for fa in solution['facilitator_assignments']:
            if fa['total_assignments'] > 0:
                f = fa['fellow']
                status = "Returning" if fa['was_facilitator_before'] else "New"
                print(f"  Fellow #{f['id']} ({status}): Primary={fa['primary_teams']}, Secondary={fa['secondary_teams']}")

        unassigned_fellows = [fa for fa in solution['facilitator_assignments'] if fa['total_assignments'] == 0]
        if unassigned_fellows:
            print(f"\n  Unassigned Fellows: {len(unassigned_fellows)}")
            for fa in unassigned_fellows:
                f = fa['fellow']
                status = "Returning" if fa['was_facilitator_before'] else "New"
                print(f"    Fellow #{f['id']} ({status})")

        # Violations
        viol = solution['constraint_violations']
        if viol['hard']:
            print(f"\n{'HARD CONSTRAINT VIOLATIONS':^80}")
            print("-" * 80)
            for v in viol['hard']:
                print(f"  ⚠ {v}")

        if viol['soft']:
            print(f"\n{'SOFT CONSTRAINT NOTES':^80}")
            print("-" * 80)
            for v in viol['soft']:
                print(f"  ℹ {v}")

        # Friend pairs
        if solution['friend_pairs_satisfied']:
            print(f"\n{'FRIEND PAIRS PLACED TOGETHER':^80}")
            print("-" * 80)
            for p1, p2, tid in solution['friend_pairs_satisfied']:
                print(f"  ✓ Participants {p1} & {p2} → Team {tid}")

        if solution['friend_pairs_unsatisfied']:
            print(f"\n{'FRIEND PAIRS NOT PLACED TOGETHER':^80}")
            print("-" * 80)
            for p1, p2 in solution['friend_pairs_unsatisfied']:
                print(f"  ✗ Participants {p1} & {p2}")

        # Unassigned
        if solution['unassigned_participants']:
            print(f"\n{'UNASSIGNED PARTICIPANTS':^80}")
            print("-" * 80)
            for p in solution['unassigned_participants'][:20]:
                total = p['total_available'] + p['total_available_if_necessary']
                print(f"  ID {p['id']}: {total} slots, Format: {p['format_pref']}, Round: {p['round_type']}, Cat: {p['category']}")
            if len(solution['unassigned_participants']) > 20:
                print(f"  ... and {len(solution['unassigned_participants']) - 20} more")

        print("\n" + "=" * 80)

    def export_solution(self, solution: dict, output_path: str = "team_assignments_v3.xlsx"):
        """Export solution to Excel with multiple sheets."""
        self.log(f"Exporting solution to {output_path}...")

        # ---- Team Assignments Sheet ----
        assignment_rows = []
        for tid, team in solution['teams'].items():
            for m in team['members']:
                round_name = {'B': 'Both Rounds', 'C': 'First Round', 'D': 'Second Round'}[team['info']['round_type']]
                assignment_rows.append({
                    'Team': tid,
                    'Day': team['info']['day'],
                    'Time': team['info']['time'],
                    'Format': 'Virtual' if team['info']['is_virtual'] else 'In-Person',
                    'Round': round_name,
                    'Participant ID': m['id'],
                    'Is Facilitator': False,
                    'Facilitator Role': '',
                    'Category': m['category'],
                    'Is Student': m['is_student'],
                    'Year': m.get('year', ''),
                    'Gender': self.GENDER_MAP.get(m.get('gender_code', 0), ''),
                    'Age Range': m.get('age_range', ''),
                    'Ideology': m.get('ideology', ''),
                    'Races': ', '.join(m.get('races', [])),
                    'Pro Liberty': m.get('issue1_position', ''),
                    'Pro Rule of Law': m.get('issue2_position', ''),
                    'Source': m.get('source', ''),
                    'Format Preference': m.get('format_preference_full', ''),
                    'Rounds Preference': m.get('rounds_full', ''),
                    'Assigned to Preferred Slot': m.get('assigned_to_preferred', True),
                })

        df_assignments = pd.DataFrame(assignment_rows)

        # ---- Team Summary Sheet ----
        summary_rows = []
        for tid, team in sorted(solution['teams'].items()):
            c = team['composition']
            round_name = {'B': 'Both Rounds', 'C': 'First Round', 'D': 'Second Round'}[team['info']['round_type']]
            pf = team['primary_facilitator']
            sf = team['secondary_facilitator']
            summary_rows.append({
                'Team': tid,
                'Day': team['info']['day'],
                'Time': team['info']['time'],
                'Format': 'Virtual' if team['info']['is_virtual'] else 'In-Person',
                'Round': round_name,
                'Size': team['size'],
                'Primary Facilitator': pf['id'] if pf else '',
                'Primary Fellow Status': 'New' if pf and not pf['was_facilitator_before'] else 'Returning' if pf else '',
                'Secondary Facilitator': sf['id'] if sf else '',
                'Secondary Fellow Status': 'New' if sf and not sf['was_facilitator_before'] else 'Returning' if sf else '',
                'Students': c['students'],
                'Non-Students': c['non_students'],
                'Women': c['women'],
                'Men': c['men'],
                'Non-binary': c.get('nonbinary', 0),
                'Conservatives': c['conservatives'],
                'Liberals': c['liberals'],
                'Moderates': c['moderates'],
                'White': c['white'],
                'Non-White': c['non_white'],
                'Pro Liberty Agree': c['issue1_agree'],
                'Pro Liberty Disagree': c['issue1_disagree'],
                'Pro Rule of Law Agree': c['issue2_agree'],
                'Pro Rule of Law Disagree': c['issue2_disagree'],
            })
        df_summary = pd.DataFrame(summary_rows)

        # ---- Facilitators Sheet ----
        fac_rows = []
        for fa in solution['facilitator_assignments']:
            f = fa['fellow']
            fac_rows.append({
                'Fellow ID': f['id'],
                'Fellow Type Code': f.get('fellow_type_code', ''),
                'Category': f.get('category', ''),
                'Gender': self.GENDER_MAP.get(f.get('gender_code', 0), ''),
                'Ideology': f.get('ideology', ''),
                'Was Facilitator Before': fa['was_facilitator_before'],
                'Was Online Last Semester': fa.get('was_online_last_semester', False),
                'Was Not Primary Last Semester': fa.get('was_not_primary_last_semester', False),
                'Primary Teams': ', '.join(str(t) for t in fa['primary_teams']),
                'Secondary Teams': ', '.join(str(t) for t in fa['secondary_teams']),
                'Total Assignments': fa['total_assignments'],
            })
        df_facilitators = pd.DataFrame(fac_rows)

        # Write to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Team Summary', index=False)
            df_assignments.to_excel(writer, sheet_name='Team Assignments', index=False)
            df_facilitators.to_excel(writer, sheet_name='Facilitators', index=False)

        self.log(f"Solution exported to {output_path}")


def main():
    """Main entry point for standalone execution."""
    data_path = "new_data/Spring D Team Data_Deidentified 2.5.26.csv"

    if not os.path.exists(data_path):
        # Try relative paths
        for alt in ["Spring D Team Data_Deidentified 2.5.26.csv",
                     "new_data/Spring D Team Data_Deidentified 2.5.26.csv"]:
            if os.path.exists(alt):
                data_path = alt
                break
        else:
            print(f"Error: Data file not found: {data_path}")
            return

    solver = DTeamSolverV3(data_path, verbose=True)

    solution = solver.solve(
        min_team_size=8,
        max_team_size=10,
        allow_flexible_size=True,
        time_limit_seconds=300,
        prioritize_low_availability=True
    )

    solver.print_report(solution)
    solver.export_solution(solution, "team_assignments_v3.xlsx")
    print("\nComplete! Check 'team_assignments_v3.xlsx' for the full solution.")


if __name__ == "__main__":
    main()
