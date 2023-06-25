from src.app.app_container import AppContainer

from src.utils.globals import Globals

import logging
import os

def configure_logger():
    # Create artifacts folder if not exists
    os.makedirs(Globals.artifacts_path, exist_ok=True)

    # Create log file
    log_file = Globals.artifacts_path.joinpath("logs.txt")
    open(log_file, "a+")

    # Configure logger
    logging.basicConfig(filename=log_file, format='[%(asctime)s][%(levelname)s] %(message)s', level=logging.INFO)

def main(app_container: AppContainer):
    configure_logger()

    app_container.trainer.train()
    app_container.trainer.test()


if __name__ == "__main__":
    app_container = AppContainer()
    main(app_container)