"""Quick test: run solver and print only key results."""
import sys
sys.path.insert(0, '.')
from dteam_solver_v3 import DTeamSolverV3

solver = DTeamSolverV3('new_data/Spring D Team Data_Deidentified 2.5.26.csv', verbose=False)

solution = solver.solve(
    min_team_size=8,
    max_team_size=10,
    allow_flexible_size=True,
    time_limit_seconds=120,
    prioritize_low_availability=True
)

# Key checks
stats = solution['statistics']
print("="*60)
print("KEY RESULTS:")
print("="*60)
print(f"Status: {solution['status']}")
print(f"Assignment rate: {stats['assignment_rate']:.1f}%")
print(f"Assigned: {stats['assigned']}/{stats['total_participants']}")
print(f"Unassigned: {stats['unassigned']}")
print(f"Teams formed: {stats['teams_formed']}")
print(f"Fellows total: {stats['total_fellows']}")
print(f"Fellows assigned: {stats['fellows_assigned']}")
print(f"New fellows assigned: {stats['new_fellows_assigned']}")
print(f"Returning fellows assigned: {stats['returning_fellows_assigned']}")
print(f"Preferred slots: {stats['assigned_to_preferred_slot']}")
print(f"If-necessary slots: {stats['assigned_to_if_necessary_slot']}")

# Check hard violations
viol = solution['constraint_violations']
print(f"\nHard violations: {len(viol['hard'])}")
for v in viol['hard']:
    print(f"  ⚠ {v}")
print(f"Soft violations: {len(viol['soft'])}")
for v in viol['soft'][:10]:
    print(f"  ℹ {v}")
if len(viol['soft']) > 10:
    print(f"  ... and {len(viol['soft']) - 10} more")

# Check fellow 163 (not primary last semester)
print("\n--- SPECIAL FELLOWS ---")
for fa in solution['facilitator_assignments']:
    f = fa['fellow']
    if f.get('was_not_primary_last_semester'):
        print(f"Fellow {f['id']} (NotPrimary last sem): Primary={fa['primary_teams']}, Secondary={fa['secondary_teams']}")
    if f.get('was_online_last_semester'):
        teams = fa['primary_teams'] + fa['secondary_teams']
        formats = []
        for t_id in teams:
            if t_id in solution['teams']:
                is_virt = solution['teams'][t_id]['info']['is_virtual']
                formats.append('Virtual' if is_virt else 'In-Person')
        print(f"Fellow {f['id']} (Online last sem): Teams={teams}, Formats={formats}")

# Team sizes
sizes = sorted([team['size'] for team in solution['teams'].values()])
print(f"\nTeam sizes: {sizes}")
print(f"Min: {min(sizes)}, Max: {max(sizes)}, Avg: {sum(sizes)/len(sizes):.1f}")

# Unassigned details
if solution['unassigned_participants']:
    print(f"\n--- UNASSIGNED ({len(solution['unassigned_participants'])}) ---")
    for p in solution['unassigned_participants']:
        total = p['total_available'] + p['total_available_if_necessary']
        print(f"  ID={p['id']}, cat={p['category']}, round={p['round_type']}, fmt={p['format_pref']}, avail={total}")
