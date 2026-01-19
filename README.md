# D-Team Formation System

**Deliberative Citizenship Initiative — Davidson College**

An automated team formation system that uses Mixed Integer Linear Programming (MILP) to optimally assign participants to deliberation teams while satisfying complex demographic and scheduling constraints.

## Overview

The D-Team Formation System helps organize participants into balanced discussion teams for the Deliberative Citizenship Initiative. It considers multiple factors including:

- **Scheduling constraints** — Participant availability across 30+ time slots
- **Format preferences** — In-person, virtual, or either
- **Demographics** — Student/non-student, gender, race/ethnicity
- **Political diversity** — Ideology, issue positions
- **Social requests** — Friend pair requests
- **Academic credit** — Course enrollment tracking

## Features

- 🧮 **MILP Optimization** — Uses PuLP solver for optimal team assignments
- 🎯 **Hard & Soft Constraints** — Guarantees critical rules while optimizing for preferences
- 🌐 **Web Interface** — Modern, responsive UI for uploading data and viewing results
- 📊 **Rich Data Display** — Comprehensive participant profiles with all available data
- 📥 **Export Options** — Download results as Excel or CSV

## Project Structure

```
d-team/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── dteam_solver_v2.py       # Core MILP solver
├── webapp/                   # Flask web application
│   ├── app.py               # Flask backend
│   ├── templates/
│   │   └── index.html       # Frontend UI
│   └── uploads/             # Temporary file storage
└── data/                     # Sample data files (optional)
    └── Sample DCI Registrant Data.xlsx
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone or download** this repository

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web application**:
   ```bash
   cd webapp
   python app.py
   ```

4. **Open your browser** to http://127.0.0.1:5000

## Usage

### Web Interface

1. **Upload** your registrant data Excel file
2. **Configure** solver parameters (team size, time limit)
3. **Run** the optimization
4. **View** team assignments with full participant details
5. **Download** results as Excel or CSV

### Command Line (Advanced)

```python
from dteam_solver_v2 import DTeamSolverV2

# Initialize solver with data
solver = DTeamSolverV2('path/to/registrant_data.xlsx')

# Run optimization
solution = solver.solve(
    min_team_size=8,
    max_team_size=10,
    time_limit_seconds=300
)

# Export results
solver.export_solution(solution, 'team_assignments.xlsx')
```

## Constraints

### Hard Constraints (Must be satisfied)

1. Every team has exactly 2 DCI Fellows (1 Primary + 1 Secondary)
2. Participants only assigned to times they're available
3. Virtual-only participants → virtual teams only
4. In-person only participants → in-person teams only
5. Every team has ≥2 students
6. Every team has ≥2 non-students

### Soft Constraints (Optimized by priority)

1. Team size: 8-10 participants (7 and 11 acceptable if necessary)
2. Fellow assignment balance
3. Either-format participants prefer in-person
4. Friend pairs placed together
5. ≥2 women per team
6. ≥2 men per team
7. ≥1 person agreeing with each issue position
8. ≥1 person disagreeing with each issue position
9. ≥1 conservative per team
10. ≥1 non-white per team
11. ≥1 white per team
12. ≥1 liberal per team

## Data Format

The input Excel file should contain the following columns:

| Column | Description |
|--------|-------------|
| `Unique ID` | Participant identifier |
| `Status` | Registration status (Confirmed/Registered) |
| `student` | Davidson Student indicator |
| `year` | Class year (for students) |
| `male`, `female`, `gennon` | Gender indicators |
| `age` | Age range |
| `ideo` | Political ideology |
| `black`, `hispanic`, `white`, `asian`, `native` | Race/ethnicity indicators |
| `immp`, `presp` | Issue position responses |
| `format` | Meeting format preference |
| `m1030`, `m1230`, ... | Availability for each time slot (1=available) |
| `Course 1`, `Course 2`, `Course 3` | Course credit indicators (1=yes) |
| `FriendInvited`, `FriendInvitedBy` | Friend pairing requests |
| `Fellow Role`, `Fellow Assignment` | For DCI Fellows |

## Technology Stack

- **Backend**: Python, Flask
- **Solver**: PuLP (MILP optimization)
- **Data Processing**: pandas, openpyxl
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Design**: Custom vintage-minimalist aesthetic

## License

Internal use only — Davidson College Deliberative Citizenship Initiative
