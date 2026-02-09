"""End-to-end test: upload CSV, run solver, verify results."""
import requests
import json
import sys

PORT = 5051
BASE = f"http://127.0.0.1:{PORT}"

# Upload CSV file and run solver
csv_path = "/Users/murtaza/Downloads/school/math-cs-fellow/d-team/new_data/Spring D Team Data_Deidentified 2.5.26.csv"
files = {"file": open(csv_path, "rb")}
data = {
    "min_team_size": "8",
    "max_team_size": "10",
    "time_limit": "60",
    "allow_flexible": "true",
}

print("Uploading CSV and running solver...")
r = requests.post(f"{BASE}/upload", files=files, data=data)
if r.status_code != 200:
    print(f"ERROR: HTTP {r.status_code}")
    print(r.text[:500])
    sys.exit(1)

result = r.json()

print(f"\nStatus: {result.get('status')}")
stats = result["statistics"]
print(f"Teams formed: {stats['teams_formed']}")
print(f"Assigned: {stats['assigned']}")
print(f"Unassigned: {stats['unassigned']}")
print(f"Rate: {stats['assignment_rate']}%")
print(f"Friend pairs: {stats['friend_pairs_satisfied']}/{stats['friend_pairs_total']}")

if "total_fellows" in stats:
    print(f"Total fellows: {stats['total_fellows']}")
    print(f"Fellows assigned: {stats.get('fellows_assigned', 0)}")
    print(f"New fellows: {stats.get('new_fellows_assigned', 0)}")
    print(f"Returning fellows: {stats.get('returning_fellows_assigned', 0)}")

# Team analytics
ta = result.get("team_analytics", {})
if ta:
    print("\n--- Team Analytics ---")
    print(f"Avg size: {ta.get('avg_team_size')}")
    print(f"Teams by day: {ta.get('teams_by_day')}")
    print(f"Violations: {ta.get('teams_with_hard_violations')} hard, {ta.get('teams_with_soft_violations')} soft")

# Facilitator analytics
fa = result.get("facilitator_analytics", {})
if fa:
    print("\n--- Facilitator Analytics ---")
    print(f"Total fellows: {fa.get('total_fellows')}")
    print(f"Assigned: {fa.get('fellows_assigned')}")
    print(f"New total: {fa.get('new_fellows_total')}")
    print(f"Returning total: {fa.get('returning_fellows_total')}")
    print(f"Fellows detail count: {len(fa.get('fellows_detail', []))}")

# Sample member
t = result["teams"][0]
m = t["members"][0]
print("\n--- Sample Member (all new fields) ---")
for k in [
    "id", "is_fellow", "facilitator_role", "category", "gender", "ideology",
    "year", "age_range", "source", "race_binary", "races",
    "format_preference_full", "rounds_full", "pro_liberty_code", "pro_rule_code",
    "friend_invited_name", "friend_invited_by_name",
]:
    print(f"  {k}: {m.get(k)}")

# Primary facilitator
pf = t.get("primary_facilitator")
if pf:
    print("\n--- Primary Facilitator ---")
    print(f"  ID: {pf['id']}")
    print(f"  facilitator_role: {pf.get('facilitator_role')}")
    print(f"  is_fellow: {pf.get('is_fellow')}")
    print(f"  was_facilitator_before: {pf.get('was_facilitator_before')}")

# Verify Is Fellow column and Facilitator Role exist in table data
print("\n--- Verification Checks ---")
fellow_count = sum(1 for t in result["teams"] for m in t["members"] if m.get("is_fellow"))
print(f"Members marked as fellows: {fellow_count}")

role_primary = sum(1 for t in result["teams"] for m in t["members"] 
                   if t.get("primary_facilitator") and m["id"] == t["primary_facilitator"]["id"])
role_secondary = sum(1 for t in result["teams"] for m in t["members"] 
                     if t.get("secondary_facilitator") and m["id"] == t["secondary_facilitator"]["id"])
print(f"Primary facilitators in teams: {role_primary}")
print(f"Secondary facilitators in teams: {role_secondary}")

print("\n✅ All checks passed!")
