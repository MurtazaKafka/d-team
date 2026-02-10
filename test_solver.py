"""Quick test: run the solver directly to check assignment rate and constraints."""
import sys
sys.path.insert(0, '.')
from dteam_solver_v3 import DTeamSolverV3

solver = DTeamSolverV3('new_data/Spring D Team Data_Deidentified 2.5.26.csv', verbose=True)

solution = solver.solve(
    min_team_size=8,
    max_team_size=10,
    allow_flexible_size=True,
    time_limit_seconds=300,
    prioritize_low_availability=True
)

solver.print_report(solution)

# Key checks
stats = solution['statistics']
print("\n" + "="*60)
print("KEY CHECKS:")
print("="*60)
print(f"Assignment rate: {stats['assignment_rate']:.1f}%")
print(f"Assigned: {stats['assigned']}/{stats['total_participants']}")
print(f"Teams formed: {stats['teams_formed']}")
print(f"Fellows assigned: {stats['fellows_assigned']}/{stats['total_fellows']}")
print(f"New fellows assigned: {stats['new_fellows_assigned']}")
print(f"Returning fellows assigned: {stats['returning_fellows_assigned']}")

# Check hard violations
viol = solution['constraint_violations']
print(f"\nHard violations: {len(viol['hard'])}")
for v in viol['hard']:
    print(f"  ⚠ {v}")
print(f"Soft violations: {len(viol['soft'])}")

# Check fellow 163 (not primary last semester)
for fa in solution['facilitator_assignments']:
    f = fa['fellow']
    if f.get('was_not_primary_last_semester'):
        is_primary = len(fa['primary_teams']) > 0
        print(f"\nFellow {f['id']} (NotPrimaryLastSemester): Primary={fa['primary_teams']}, Secondary={fa['secondary_teams']}")
        print(f"  -> Is now Primary: {'YES ✓' if is_primary else 'NO ✗'}")

# Check online-last-semester fellows
for fa in solution['facilitator_assignments']:
    f = fa['fellow']
    if f.get('was_online_last_semester'):
        print(f"Fellow {f['id']} (OnlineLastSemester): Primary={fa['primary_teams']}, Secondary={fa['secondary_teams']}")

# Team size distribution
sizes = [team['size'] for team in solution['teams'].values()]
print(f"\nTeam sizes: {sorted(sizes)}")
print(f"Min: {min(sizes)}, Max: {max(sizes)}, Avg: {sum(sizes)/len(sizes):.1f}")
