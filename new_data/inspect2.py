import pandas as pd
df = pd.read_csv("/Users/murtaza/Downloads/school/math-cs-fellow/d-team/new_data/Spring D Team Data_Deidentified 2.5.26.csv")
data = df.iloc[2:].copy()

# Friend data
print("=== Friend Invited (has ID) ===")
for idx, row in data.iterrows():
    fi = row.get("Unnamed: 129", "")
    if pd.notna(fi) and str(fi).strip() and str(fi) != "FriendInvited":
        uid = row.get("Unnamed: 0", "?")
        i1 = row.get("Invited_1", "")
        i2 = row.get("Invited_2", "")
        print(f"  ID={uid}: FriendInvited(ID)={fi}, Names={i1} {i2}")

print("\n=== Invited By (has ID) ===")
for idx, row in data.iterrows():
    fib = row.get("Unnamed: 130", "")
    if pd.notna(fib) and str(fib).strip() and str(fib) != "FriendInvitedBy":
        uid = row.get("Unnamed: 0", "?")
        b1 = row.get("Invited by_1", "")
        b2 = row.get("Invited by_2", "")
        print(f"  ID={uid}: FriendInvitedBy(ID)={fib}, Names={b1} {b2}")

print("\n=== Q9 (Age) ===")
vals = data["Q9"].dropna()
vals = vals[vals != "Age"]
print(vals.value_counts().to_dict())

print("\n=== Q6 (Source) ===")
vals = data["Q6"].dropna()
vals = vals[vals != "Source"]
print(vals.value_counts().to_dict())

print("\n=== Q7 (Year) ===")
vals = data["Q7"].dropna()
vals = vals[vals != "Year in College"]
print(vals.value_counts().to_dict())

print("\n=== Unnamed: 25 (Race binary) ===")
vals = data["Unnamed: 25"].dropna()
vals = vals[vals != "Race"]
print(vals.value_counts().to_dict())
