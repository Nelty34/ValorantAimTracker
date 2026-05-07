import json
from scipy.stats import ttest_ind

# Load data
with open("users.json", "r") as f:
    data = json.load(f)

test_group = ["Shortbread", "Natt", "Nut"]
control_group = ["Netly", "Speegy", "Nucleic"]

def get_improvement(user, metric):
    matches = [m for m in data[user]["matches"] if m]
    if len(matches) < 2:
        return None
    return matches[-1].get(metric, 0) - matches[0].get(metric, 0)

def run_test(metric_name, func):
    test_vals = [func(u) for u in test_group]
    control_vals = [func(u) for u in control_group]

    test_vals = [v for v in test_vals if v is not None]
    control_vals = [v for v in control_vals if v is not None]

    t_stat, p_value = ttest_ind(test_vals, control_vals, equal_var=False)

    print(f"\n--- {metric_name} ---")
    print(f"Test group improvements: {test_vals}")
    print(f"Control group improvements: {control_vals}")
    print(f"T-statistic: {round(t_stat, 4)}")
    print(f"P-value: {round(p_value, 4)}")

# Run tests
run_test("ACS Improvement", lambda u: get_improvement(u, "acs"))
run_test("Kills Improvement", lambda u: get_improvement(u, "kills"))
run_test("Reaction Time Improvement", lambda u: get_improvement(u, "avg_reaction_time"))