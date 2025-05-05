import logging
import re
import os
from pathlib import Path

def val_dir():
    base_path = os.getcwd() + "/run/val"
    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True)  # Creates 'run/val' if missing

    # Find existing valX folders
    existing_folders = [d for d in os.listdir(base_dir) if d.startswith("val") and d[3:].isdigit()]
    
    # Extract numbers and determine the next available one
    existing_numbers = sorted([int(d[3:]) for d in existing_folders]) if existing_folders else []
    next_number = existing_numbers[-1] + 1 if existing_numbers else 1

    # Create new directory
    new_folder = base_dir / f"val{next_number}"
    new_folder.mkdir()
    
    return new_folder


def inference_dir():
    base_path = os.getcwd() + "/run/inference"
    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True)  # Creates 'run/inference' if missing

    # Find existing valX folders
    existing_folders = [d for d in os.listdir(base_dir) if d.startswith("run") and d[3:].isdigit()]
    
    # Extract numbers and determine the next available one
    existing_numbers = sorted([int(d[3:]) for d in existing_folders]) if existing_folders else []
    print(existing_folders)
    next_number = existing_numbers[-1] + 1 if existing_numbers else 1

    # Create new directory
    new_folder = base_dir / f"run{next_number}"
    new_folder.mkdir()
    
    return new_folder


def vid_dir():
    base_path = os.getcwd() + "/run/vid"
    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True)  # Creates 'run/vid' if missing

    # Find existing valX folders
    existing_folders = [d for d in os.listdir(base_dir) if d.startswith("vid") and d[3:].isdigit()]
    
    # Extract numbers and determine the next available one
    existing_numbers = sorted([int(d[3:]) for d in existing_folders]) if existing_folders else []
    next_number = existing_numbers[-1] + 1 if existing_numbers else 1

    # Create new directory
    new_folder = base_dir / f"vid{next_number}"
    new_folder.mkdir()
    
    return new_folder

# Create a formatter that strips ANSI color codes
class ColorStrippingFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        # Regex to match ANSI escape sequences
        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    
    def format(self, record):
        # Format the message
        formatted_message = super().format(record)
        # Strip ANSI escape sequences
        return self.ansi_escape.sub('', formatted_message)

# Set up your loggers
def setup_logger(output_dir):
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler (keeps colors)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')  # Simple format for console
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (strips colors)
    file_handler = logging.FileHandler(f'{output_dir}/val.log')
    file_formatter = ColorStrippingFormatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

