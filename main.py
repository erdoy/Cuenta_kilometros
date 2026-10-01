import argparse
import re
from pathlib import Path

import pandas as pd
from tabulate import tabulate


MESSAGE_PATTERN = re.compile(
	r"^(?P<date>\d{1,2}\.\d{1,2}\.\d{2,4}), "
	r"(?P<time>\d{1,2}:\d{2}) - (?P<sender>[^:]+):"
)
MESSAGE_BOUNDARY_PATTERN = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{2,4}, \d{1,2}:\d{2} - ")
FIELD_PATTERN = re.compile(r"^\s*(?P<key>[^:]+?)\s*:\s*(?P<value>.*?)\s*$")


def extract_trips(chat_text: str) -> pd.DataFrame:
	"""Extract WhatsApp messages followed by one or more key-value fields."""
	records = []
	current_record = None

	def save_current_record():
		if current_record is not None and current_record["fields"]:
			records.append({**current_record["message"], **current_record["fields"]})

	for line in chat_text.splitlines():
		message_match = MESSAGE_PATTERN.match(line)
		if message_match:
			save_current_record()
			current_record = {
				"message": message_match.groupdict(),
				"fields": {},
			}
			continue

		if MESSAGE_BOUNDARY_PATTERN.match(line):
			save_current_record()
			current_record = None
			continue

		if current_record is None:
			continue

		field_match = FIELD_PATTERN.match(line)
		if field_match:
			key = field_match.group("key").strip()
			value = field_match.group("value").strip()
			if key and value:
				current_record["fields"][key] = value

	save_current_record()
	table = pd.DataFrame(records)
	if "Km" in table.columns:
		table["Km"] = pd.to_numeric(table["Km"], errors="coerce")
	return table


def _normalize_value(value):
	return str(value).strip().casefold() if pd.notna(value) else ""


def _build_owner_groups(car_owners: pd.DataFrame):
	required_columns = {"Integrantes", "Coche"}
	if not required_columns.issubset(car_owners.columns):
		raise ValueError("The car-owners CSV must have Integrantes and Coche columns.")

	people = {}
	cars = {}
	people_by_car = {}

	for _, row in car_owners.iterrows():
		person_key = _normalize_value(row["Integrantes"])
		car_key = _normalize_value(row["Coche"])
		if not person_key or not car_key:
			continue
		people.setdefault(person_key, str(row["Integrantes"]).strip())
		cars.setdefault(car_key, str(row["Coche"]).strip())
		people_by_car.setdefault(car_key, set()).add(person_key)

	parents = {person_key: person_key for person_key in people}

	def find(person_key):
		while parents[person_key] != person_key:
			parents[person_key] = parents[parents[person_key]]
			person_key = parents[person_key]
		return person_key

	for car_people in people_by_car.values():
		car_people = iter(car_people)
		first_person = next(car_people)
		for person_key in car_people:
			parents[find(person_key)] = find(first_person)

	groups = {}
	owner_group_by_person = {}
	for person_key, person_name in people.items():
		group_key = find(person_key)
		owner_group_by_person[person_key] = group_key
		group = groups.setdefault(group_key, {"people": set(), "cars": set()})
		group["people"].add(person_name)
	owner_group_by_car = {}
	for car_key, car_people in people_by_car.items():
		group_key = find(next(iter(car_people)))
		groups[group_key]["cars"].add(car_key)
		owner_group_by_car[car_key] = group_key
	return owner_group_by_car, owner_group_by_person, groups, cars


def compare_owner_kilometers(trips: pd.DataFrame, car_owners: pd.DataFrame) -> pd.DataFrame:
	"""Sum trip kilometers by connected groups of people who share cars."""
	owner_group_by_car, _, groups, cars = _build_owner_groups(car_owners)
	trip_cars = trips["Coche"].map(_normalize_value)

	summary = []
	for group in groups.values():
		group_cars = group["cars"]
		kilometers = trips.loc[trip_cars.isin(group_cars), "Km"].sum()
		summary.append(
			{
				"Integrantes": ", ".join(sorted(group["people"])),
				"Coches": ", ".join(sorted(cars[car] for car in group_cars)),
				"Km": kilometers,
			}
		)

	return pd.DataFrame(summary, columns=["Integrantes", "Coches", "Km"]).sort_values(
		"Km", ascending=False, ignore_index=True
	)


def calculate_car_kilometer_debts(trips: pd.DataFrame, car_owners: pd.DataFrame) -> pd.DataFrame:
	"""Net kilometers owed between trip participants, split by owner group."""
	trip_columns = {"Integrantes", "Coche", "Km"}
	if not trip_columns.issubset(trips.columns):
		raise ValueError("Trips must have Integrantes, Coche, and Km columns.")

	owner_group_by_car, owner_group_by_person, groups, _ = _build_owner_groups(car_owners)
	group_names = {
		group_key: ", ".join(sorted(group["people"]))
		for group_key, group in groups.items()
	}
	gross_debts = {}
	unmapped_cars = set()

	for _, trip in trips.iterrows():
		car_key = _normalize_value(trip["Coche"])
		if not car_key:
			continue
		owner_group = owner_group_by_car.get(car_key)
		if owner_group is None:
			unmapped_cars.add(str(trip["Coche"]).strip())
			continue
		if pd.isna(trip["Km"]) or pd.isna(trip["Integrantes"]):
			continue

		participants = set(re.findall(r"[A-Z]", str(trip["Integrantes"])))
		participant_groups = {
			owner_group_by_person.get(participant.casefold(), participant.casefold())
			for participant in participants
		}
		if not participant_groups:
			continue
		kilometers_per_group = float(trip["Km"]) / len(participant_groups)
		for participant_group in participant_groups:
			if participant_group == owner_group:
				continue
			debt_key = (owner_group, participant_group)
			gross_debts[debt_key] = gross_debts.get(debt_key, 0) + kilometers_per_group

	if unmapped_cars:
		car_list = ", ".join(sorted(unmapped_cars))
		raise ValueError(f"Add these cars to the car-owners CSV before calculating debts: {car_list}")

	debts = []
	processed_pairs = set()
	for debtor_group, creditor_group in sorted(gross_debts):
		pair = frozenset((debtor_group, creditor_group))
		if pair in processed_pairs:
			continue
		processed_pairs.add(pair)
		net_debt = gross_debts.get((debtor_group, creditor_group), 0) - gross_debts.get(
			(creditor_group, debtor_group), 0
		)
		if net_debt == 0:
			continue
		if net_debt < 0:
			debtor_group, creditor_group = creditor_group, debtor_group
			net_debt = -net_debt
		debts.append(
			{
				"Owes": group_names.get(debtor_group, debtor_group.upper()),
				"Owed to": group_names.get(creditor_group, creditor_group.upper()),
				"Km owed": net_debt,
			}
		)

	return pd.DataFrame(debts, columns=["Owes", "Owed to", "Km owed"]).sort_values(
		"Km owed", ascending=False, ignore_index=True
	)


def main():
	parser = argparse.ArgumentParser(
		description="Extract trip details from an exported WhatsApp chat."
	)
	parser.add_argument("chat_file", type=Path, help="Path to the exported .txt chat")
	parser.add_argument(
		"--output-csv",
		type=Path,
		help="Optional path for saving the extracted table as CSV",
	)
	parser.add_argument(
		"--compact",
		action="store_true",
		help="Show only date, time, Integrantes, and Km",
	)
	parser.add_argument(
		"--car-owners",
		type=Path,
		help="Compare kilometers using a CSV with Integrantes and Coche columns",
	)
	parser.add_argument(
		"--equilibrium",
		action="store_true",
		help="Show net car kilometers owed between owner groups based on trip participants",
	)
	args = parser.parse_args()
	if args.equilibrium and not args.car_owners:
		parser.error("--equilibrium requires --car-owners")

	chat_text = args.chat_file.read_text(encoding="utf-8-sig")
	table = extract_trips(chat_text)
	if args.car_owners:
		car_owners = pd.read_csv(args.car_owners, dtype=str).fillna("")
		if args.equilibrium:
			display_table = calculate_car_kilometer_debts(table, car_owners)
		else:
			display_table = compare_owner_kilometers(table, car_owners)
	else:
		display_table = table.reindex(columns=["date", "time", "Integrantes", "Km"]) if args.compact else table
	print(tabulate(display_table, headers="keys", tablefmt="grid", showindex=False, floatfmt=".2f"))

	if args.output_csv:
		table.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
		print(f"\nSaved {len(table)} row(s) to {args.output_csv}")


if __name__ == "__main__":
	main()
