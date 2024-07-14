import yaml
import os
import logging
import colorlog
from icrawler.builtin import GoogleImageCrawler


search_queries = [
    "tennis player",
    "tennis face",
    "tennis athlete",
    "tennis photo",
    "tennis portrait",
]

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create a color formatter
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)

# Create a handler
handler = logging.StreamHandler()
handler.setFormatter(formatter)

# Get the root logger
logger = logging.getLogger("data_loader")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)


def download_images(
    players: list[str], num_images: int = 100, output_dir: str = "./data/player_images"
) -> None:
    for player in players:
        logger.info(f"Start downloading for player {player}...")
        player_dir = os.path.join(output_dir, player.replace(" ", "_"))
        if not os.path.exists(player_dir):
            os.makedirs(player_dir)

        google_crawler = GoogleImageCrawler(storage={"root_dir": player_dir})

        for query in search_queries:
            logger.info(f"Start query - {query} - for player {player}...")
            google_crawler.crawl(
                keyword=player + " " + query,
                max_num=num_images,
                filters={"type": "face"},
                file_idx_offset="auto",  # Ensures file names are unique and don't overwrite existing images
            )
        print(f"Downloaded images for {player}")


if __name__ == "__main__":
    # Load the list of players from the config file
    path_config = "config/data-loader.yaml"
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
        players = config["players"]

    download_images(players)
