import pandas as pd

df = pd.read_csv("/Users/murtaza/Downloads/school/math-cs-fellow/d-team/new_data/Spring D Team Data_Deidentified 2.5.26.csv")
data = df.iloc[2:].copy()
print("Total data rows:", len(data))

# Row 2 internal names
row2 = df.iloc[1]

# Check the Unnamed columns at the end - these are the new facilitator cols
for col in ["Unnamed: 129", "Unnamed: 130", "Unnamed: 131", "Unnamed: 132", "Unnamed: 133"]:
    row0_val = df.iloc[0].get(col, "N/A")
    row1_val = df.iloc[1].get(col, "N/A")
    vals = data[col].dropna()
    print(f"\n{col}:")
    print(f"  Header row 0: {repr(row0_val)}")
    print(f"  Header row 1: {repr(row1_val)}")
    print(f"  Data values ({len(vals)} non-null): {vals.value_counts().head(5).to_dict()}")

# Key demographics
print("\n=== KEY COLUMNS ===")
for col in ["Unnamed: 0", "Q5", "Q7", "Q6", "Q8", "Q9", "Q10", "Race/Ethnicity", "Unnamed: 25", "Pro Liberty", "Pro Rule of Law", "Format", "Rounds"]:
    vals = data[col].dropna()
    print(f"\n{col} ({len(vals)} non-null): {vals.value_counts().head(10).to_dict()}")

# Check friend columns
print("\n=== FRIEND COLUMNS ===")
for col in ["Invited_1", "Invited_2", "Invited_3", "Invited by_1", "Invited by_2", "Invited by_3"]:
    vals = data[col].dropna()
    print(f"{col} ({len(vals)} non-null)")

# Facilitator columns at the end
# From the head output, row 3 has: FriendInvited, FriendInvitedBy, Facilitator, Online Last Semester, Not Primary Last Semester
# These are at Unnamed: 129-133
print("\n=== BOTTOM ROW NAMES (row index 1 = 3rd CSV row) ===")
r = df.iloc[1]
for i in range(128, 134):
    col = df.columns[i] if i < len(df.columns) else "N/A"
    print(f"  col[{i}] = {repr(col)}, row1 val = {repr(r.get(col, 'N/A'))}")
