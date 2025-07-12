import logging

def setup_logging(log_file="training.log"):
    """
    Configures the logging system to output messages to both the console and a file.

    The log file will be created in the current working directory.
    The default logging level is INFO, and messages are formatted with
    timestamp, level, and the log message.

    Args:
        log_file (str): The name of the file to which log messages will be written.
                        Defaults to "training.log".
    """
    logging.basicConfig(
        # Set the minimum level for messages to be processed
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    # Ensure the root logger level is set to INFO to process messages
    logging.getLogger().setLevel(logging.INFO) 

if __name__ == '__main__':
    setup_logging()
    logging.info("Logging setup complete.")
