import logging
import os
from configs.config import LOGS_DIR


logging.basicConfig(
    filename=os.path.join(LOGS_DIR, 'app.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info("all is ok")
