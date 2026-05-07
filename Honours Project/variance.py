import json
import statistics

INPUT_FILE = "users.json"


def load_data():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_values(matches, key):
    values = []

    for match in matches:
        value = match.get(key)

        if isinstance(value, (int, float)):
            values.append(value)

    return values


def average(values):
    return round(sum(values) / len(values), 3) if values else 0


def value_range(values):
    if not values:
        return 0
    return round(max(values) - min(values), 3)


def standard_deviation(values):
    if len(values) < 2:
        return 0
    return round(statistics.stdev(values), 3)


def population_variance(values):
    if len(values) < 2:
        return 0
    return round(statistics.pvariance(values), 3)


def list_users(data):
    print("\nUsers:")
    users = list(data.keys())

    for i, user in enumerate(users, 1):
        print(f"{i}. {user}")

    return users


def choose_user(data):
    users = list_users(data)

    while True:
        try:
            choice = int(input("\nSelect user number: "))
            if 1 <= choice <= len(users):
                return users[choice - 1]
        except ValueError:
            pass

        print("Invalid user selection.")


def list_matches(matches):
    print("\nMatches:")

    for i, match in enumerate(matches, 1):
        map_name = match.get("map", "Unknown")
        scoreline = match.get("scoreline", "No score")
        kills = match.get("kills", 0)
        deaths = match.get("deaths", 0)
        assists = match.get("assists", 0)

        print(f"{i}. {map_name} | {scoreline} | KDA {kills}/{deaths}/{assists}")


def choose_matches(matches):
    list_matches(matches)

    print("\nEnter match numbers separated by commas.")
    print("Example: 1,2,4")

    while True:
        raw = input("\nSelect matches: ")

        try:
            selected_indexes = [
                int(x.strip()) - 1
                for x in raw.split(",")
                if x.strip()
            ]

            if selected_indexes and all(0 <= i < len(matches) for i in selected_indexes):
                return [matches[i] for i in selected_indexes]

        except ValueError:
            pass

        print("Invalid match selection.")


def analyse_matches(selected_matches):
    metrics = [
        "kills",
        "deaths",
        "assists",
        "hs_percent",
        "acs",
        "avg_reaction_time",
        "total_engagements",
        "engagements_with_shots"
    ]

    print("\nVariation Across Selected Matches")
    print("--------------------------------")

    for metric in metrics:
        values = safe_values(selected_matches, metric)

        print(f"\n{metric}")
        print(f"  values: {values}")
        print(f"  average: {average(values)}")
        print(f"  range: {value_range(values)}")
        print(f"  standard deviation: {standard_deviation(values)}")
        print(f"  population variance: {population_variance(values)}")


def main():
    data = load_data()

    username = choose_user(data)
    matches = data[username].get("matches", [])

    if not matches:
        print("No matches found for this user.")
        return

    selected_matches = choose_matches(matches)

    print(f"\nUser Selected: {username}")
    print(f"Matches Analysed: {len(selected_matches)}")

    analyse_matches(selected_matches)


if __name__ == "__main__":
    main()