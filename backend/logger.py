import logging
import sys

app_log = logging.getLogger("app")
analysis_log = logging.getLogger("analysis")

app_log.propagate = False
analysis_log.propagate = False

formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Handlers

file_handler = logging.FileHandler("logs.log")
stream_handler = logging.StreamHandler(sys.stdout)

stream_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


app_log.handlers = [stream_handler]
analysis_log.handlers = [file_handler]


app_log.setLevel(logging.INFO)
analysis_log.setLevel(logging.INFO)
