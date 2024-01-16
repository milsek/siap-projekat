import logging
import json
import csv
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(message)s",
)

fields = {
    "object_price": "cena",
    "object_brand": "marka",
    "object_model": "model",
    "object_production_year": "godina proizvodnje",
    "object_mileage": "kilometraža",
    "object_chassis": "karoserija",
    "object_fuel": "gorivo",
    "object_engine_volume": "kubikaža",
    "object_engine_horsepower": "snaga motora",
    "object_gear_box": "menjač",
    "object_door_num": "broj vrata",
    "object_air_conditioner": "klima",
}

folder_path = "vehicles"

csv_file_path = "vehicles.csv"

with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields.values())
    writer.writeheader()

converted = 0
total_vehicles = 58975  # ls -l vehicles/ | grep ^- | wc -l

for file_name in os.listdir(folder_path):
    if file_name.endswith(".json"):
        with open(
            os.path.join(folder_path, file_name), "r", encoding="utf-8"
        ) as json_file:
            data = json.load(json_file)
            with open(csv_file_path, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=fields.values(), quoting=csv.QUOTE_MINIMAL
                )
                writer.writerow({fields[key]: data[key] for key in fields})
                converted += 1
                logging.info("Converted %d out of %d.", converted, total_vehicles)

logging.info("Done")
