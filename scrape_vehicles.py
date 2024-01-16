import sys
import time
import json
import logging
import requests
import threading
from queue import Queue


logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(message)s",
)

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0"
}

url_queue = Queue(maxsize=10000)


def produce_urls():
    num_of_pages = 2709

    for pagenum in range(381, num_of_pages + 1):
        start = time.time()
        urls = get_ad_urls(pagenum)
        elapsed = time.time() - start
        logging.info(
            "Fetched %d urls from page %d out of %d (%.2f%%) in %.2f seconds. In queue: %d",
            len(urls),
            pagenum,
            num_of_pages,
            (pagenum / num_of_pages) * 100,
            elapsed,
            url_queue.qsize(),
        )
        for url in urls:
            url_queue.put(url)


def consume_urls():
    while True:
        url = url_queue.get()
        if url is None:
            break

        filename = export_vehicle_data(url)
        logging.info("Exported %s", filename)
        url_queue.task_done()

    logging.info("Done")


def main():
    threading.Thread(target=produce_urls).start()

    for i in range(32):
        threading.Thread(target=consume_urls).start()


def get_ad_urls(pagenum):
    response = requests.get(
        f"https://www.polovniautomobili.com/auto-oglasi/pretraga?page={pagenum}&sort=basic&city_distance=0&showOldNew=all",
        headers=headers,
    )

    return set(
        [
            "https://www.polovniautomobili.com/auto-oglasi/" + u.split('"')[0]
            for u in response.text.split('\'});" href="/auto-oglasi/')[1:]
        ]
    )


def export_vehicle_data(url):
    repsonse = requests.get(url, headers=headers)

    filename = url.split("/auto-oglasi/")[1].split("?")[0].replace("/", "_") + ".json"
    vehicle_data = (
        '{"' + repsonse.text.split('dataLayer.push({"')[1].split("});")[0] + "}"
    )

    with open("vehicles/" + filename, "w") as file:
        vehicle_data_json = json.loads(vehicle_data)
        vehicle_data_formatted = json.dumps(vehicle_data_json, indent=4)

        file.write(vehicle_data_formatted)

    return filename


if __name__ == "__main__":
    main()
