import json
from collections import Counter

INPUT_FILE = "users.json"
OUTPUT_FILE = "users_averages.json"


def safe_avg(values):
    """Return average of numeric values."""
    values = [v for v in values if isinstance(v, (int, float))]
    return round(sum(values) / len(values), 3) if values else 0


def most_common(values):
    """Return most common string value."""
    values = [v for v in values if v]
    if not values:
        return None
    return Counter(values).most_common(1)[0][0]


def process_users():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = {}

    for username in sorted(data.keys()):
        user = data[username]
        matches = user.get("matches", [])

        if not matches:
            continue

        kills = []
        deaths = []
        assists = []
        hs_percent = []
        acs = []
        analysis_time = []
        avg_reaction_time = []
        total_engagements = []
        engagements_with_shots = []
        crosshair_positions = []

        total_matches = len(matches)

        for match in matches:
            kills.append(match.get("kills", 0))
            deaths.append(match.get("deaths", 0))
            assists.append(match.get("assists", 0))
            hs_percent.append(match.get("hs_percent", 0))
            acs.append(match.get("acs", 0))
            analysis_time.append(match.get("analysis_time", 0))
            avg_reaction_time.append(match.get("avg_reaction_time", 0))
            total_engagements.append(match.get("total_engagements", 0))
            engagements_with_shots.append(match.get("engagements_with_shots", 0))

            if match.get("avg_crosshair_placement"):
                crosshair_positions.append(match["avg_crosshair_placement"])

        output[username] = {
            "name": user.get("name"),
            "tag": user.get("tag"),
            "matches_played": total_matches,
            "avg_kills": safe_avg(kills),
            "avg_deaths": safe_avg(deaths),
            "avg_assists": safe_avg(assists),
            "avg_hs_percent": safe_avg(hs_percent),
            "avg_acs": safe_avg(acs),
            "avg_analysis_time": safe_avg(analysis_time),
            "avg_reaction_time": safe_avg(avg_reaction_time),
            "avg_total_engagements": safe_avg(total_engagements),
            "avg_engagements_with_shots": safe_avg(engagements_with_shots),
            "avg_crosshair_placement": most_common(crosshair_positions),
            "kd_ratio": round(sum(kills) / max(sum(deaths), 1), 3)
        }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    print(f"Averages saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    process_users()