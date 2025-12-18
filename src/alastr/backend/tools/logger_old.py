import os
import logging
from datetime import datetime

# Create a dedicated logger
logger = logging.getLogger("CustomLogger")
logger.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Create a stream handler (console output)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# Function to configure the file handler once output_label is known
def configure_file_handler(output_label):
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    log_path = os.path.join(os.getcwd(), "logs", f"{output_label}_{timestamp}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.info(f"Log file initialized at {log_path}.")
