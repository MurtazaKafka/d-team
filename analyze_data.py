"""Analyze the CSV data to understand demographics and constraints."""
import pandas as pd
import numpy as np

df = pd.read_csv('new_data/Spring D Team Data_Deidentified 2.5.26.csv')
data = df.iloc[2:].copy().reset_index(drop=True)

# Convert numeric columns
for col in ['Q5','Q8','Q10','Format','Rounds','Pro Liberty','Pro Rule of Law']:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# Check unnamed columns for facilitator data
unnamed = [c for c in data.columns if 'Unnamed' in str(c)]
print("=== UNNAMED COLUMNS ===")
for uc in unnamed:
    vals = data[uc].dropna()
    if len(vals) > 0:
        print(f"  {uc}: {len(vals)} values, unique={vals.unique()[:10]}")

# Rename key unnamed columns
rename_map = {}
for col in data.columns:
    if col == 'Unnamed: 0': rename_map[col] = 'UniqueID'
    elif col == 'Unnamed: 25': rename_map[col] = 'RaceBinary'
    elif col == 'Unnamed: 129': rename_map[col] = 'FriendInvited'
    elif col == 'Unnamed: 130': rename_map[col] = 'FriendInvitedBy'
    elif col == 'Unnamed: 131': rename_map[col] = 'Facilitator'
    elif col == 'Unnamed: 132': rename_map[col] = 'OnlineLastSemester'
    elif col == 'Unnamed: 133': rename_map[col] = 'NotPrimaryLastSemester'
data.rename(columns=rename_map, inplace=True)

for col in ['UniqueID','Facilitator','OnlineLastSemester','NotPrimaryLastSemester','RaceBinary','FriendInvited','FriendInvitedBy']:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

print(f"\n=== TOTAL ROWS: {len(data)} ===")

# Facilitators
if 'Facilitator' in data.columns:
    fac = data[data['Facilitator'].isin([1,2])]
    non_fac = data[~data['Facilitator'].isin([1,2])]
    print(f"\nFacilitators: {len(fac)}")
    print(f"Regular participants: {len(non_fac)}")
    
    # Check if facilitators also have Q5 (student status)
    fac_students = fac[fac['Q5'] == 1]
    fac_nonstudents = fac[fac['Q5'].isin([2,3,4,5])]
    fac_q5_missing = fac[fac['Q5'].isna()]
    print(f"  Fac who are students: {len(fac_students)}")
    print(f"  Fac who are non-students: {len(fac_nonstudents)}")
    print(f"  Fac Q5 missing: {len(fac_q5_missing)}")
else:
    non_fac = data
    print("NO Facilitator column found!")

# Student vs non-student among regular participants
students = non_fac[non_fac['Q5'] == 1]
non_students = non_fac[non_fac['Q5'].isin([2,3,4,5])]
q5_nan = non_fac[non_fac['Q5'].isna()]
print(f"\n=== STUDENT/NON-STUDENT (regular participants only) ===")
print(f"Students: {len(students)}")
print(f"Non-students: {len(non_students)}")
print(f"  Q5 missing: {len(q5_nan)}")
print(f"  Q5 value counts: {non_fac['Q5'].value_counts().to_dict()}")

# How many teams can we form?
# Each team needs >=2 students + >=2 non-students
max_teams_by_students = len(students) // 2
max_teams_by_nonstudents = len(non_students) // 2
print(f"\nMax teams by student constraint: {max_teams_by_students}")
print(f"Max teams by non-student constraint: {max_teams_by_nonstudents}")
print(f"Bottleneck: {'NON-STUDENTS' if max_teams_by_nonstudents < max_teams_by_students else 'STUDENTS'}")

# Format preferences
print(f"\n=== FORMAT PREFERENCES (regular participants) ===")
for v, label in [(1,'Virtual Only'), (2,'In-Person Only'), (3,'Either')]:
    ct = len(non_fac[non_fac['Format'] == v])
    print(f"  {label}: {ct}")
print(f"  Missing: {non_fac['Format'].isna().sum()}")

# Rounds preferences
print(f"\n=== ROUNDS PREFERENCES (regular participants) ===")
for v, label in [(1,'Both Rounds'), (2,'First Round Only'), (3,'Second Round Only')]:
    ct = len(non_fac[non_fac['Rounds'] == v])
    print(f"  {label}: {ct}")
print(f"  Missing: {non_fac['Rounds'].isna().sum()}")

# Availability analysis
print(f"\n=== AVAILABILITY ANALYSIS ===")
avail_cols_B = [c for c in data.columns if c.startswith(('Fri_','Sat_','Sun_','Mon_','Tue_','Wed_','Thu_')) and not c.startswith(('Fri1','Fri2','Sat1','Sat2','Sun1','Sun2','Mon1','Mon2','Tue1','Tue2','Wed1','Wed2','Thu1','Thu2'))]
avail_cols_C = [c for c in data.columns if c.startswith(('Fri1_','Sat1_','Sun1_','Mon1_','Tue1_','Wed1_','Thu1_'))]
avail_cols_D = [c for c in data.columns if c.startswith(('Fri2_','Sat2_','Sun2_','Mon2_','Tue2_','Wed2_','Thu2_','Q48_'))]

print(f"Both-round availability columns: {len(avail_cols_B)}")
print(f"First-round-only availability columns: {len(avail_cols_C)}")
print(f"Second-round-only availability columns: {len(avail_cols_D)}")

# For each round type, count participants with ZERO availability
for round_val, round_label, avail_cols in [(1,'Both',avail_cols_B), (2,'First Only',avail_cols_C), (3,'Second Only',avail_cols_D)]:
    round_ppl = non_fac[non_fac['Rounds'] == round_val]
    if len(round_ppl) == 0:
        continue
    zero_avail = 0
    low_avail = 0
    for _, row in round_ppl.iterrows():
        total = 0
        for c in avail_cols:
            v = pd.to_numeric(row.get(c), errors='coerce')
            if v in [1, 2]:
                total += 1
        if total == 0:
            zero_avail += 1
        elif total <= 3:
            low_avail += 1
    print(f"\n  {round_label} ({len(round_ppl)} people):")
    print(f"    Zero availability: {zero_avail}")
    print(f"    Very low (1-3 slots): {low_avail}")

# Gender breakdown
print(f"\n=== GENDER (regular participants) ===")
for v, label in [(1,'Male'),(2,'Female'),(3,'Non-binary'),(4,'Prefer not to say')]:
    ct = len(non_fac[non_fac['Q8'] == v])
    print(f"  {label}: {ct}")

# Ideology breakdown
print(f"\n=== IDEOLOGY (regular participants) ===")
for v, label in [(1,'Very Conservative'),(2,'Somewhat Conservative'),(3,'Moderate'),(4,'Somewhat Liberal'),(5,'Very Liberal'),(6,'Prefer not to say')]:
    ct = len(non_fac[non_fac['Q10'] == v])
    print(f"  {label}: {ct}")

# Race binary
if 'RaceBinary' in data.columns:
    print(f"\n=== RACE BINARY (regular participants) ===")
    for v, label in [(0,'Prefer not to respond'),(1,'Non-White'),(2,'White')]:
        ct = len(non_fac[non_fac['RaceBinary'] == v])
        print(f"  {label}: {ct}")
    print(f"  Missing: {non_fac['RaceBinary'].isna().sum()}")

# Friend pairs
if 'FriendInvited' in data.columns:
    fi = non_fac[non_fac['FriendInvited'].notna()]
    fib = non_fac[non_fac['FriendInvitedBy'].notna()]
    print(f"\n=== FRIEND PAIRS ===")
    print(f"People who invited a friend: {len(fi)}")
    print(f"People invited by a friend: {len(fib)}")

# Facilitator details
if 'Facilitator' in data.columns:
    print(f"\n=== FACILITATOR DETAILS ===")
    for _, f in fac.iterrows():
        uid = f.get('UniqueID')
        online = f.get('OnlineLastSemester')
        not_primary = f.get('NotPrimaryLastSemester')
        q5 = f.get('Q5')
        rounds = f.get('Rounds')
        fmt = f.get('Format')
        print(f"  ID={uid}, Q5={q5}, Rounds={rounds}, Format={fmt}, Online_last={online}, NotPrimary_last={not_primary}")
