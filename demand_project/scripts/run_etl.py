import pandas as pd
from DFP import AdvancedDemandForecastPipeline
from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()

HOST_CODE_PATH = Path(os.getenv("HOST_CODE_PATH"))
CONTAINER_CODE_PATH = Path(os.getenv("CONTAINER_CODE_PATH"))

DATA_PATH = HOST_CODE_PATH / 'data' / 'raw' / 'online_retail.parquet'
CLEANED =  HOST_CODE_PATH / 'data' / 'processed'
OUTPUT_DIR = HOST_CODE_PATH / 'reports' / 'figures'

df = pd.read_parquet(DATA_PATH)

# etl
pipeline = AdvancedDemandForecastPipeline(df, output_dir=CLEANED)
pipeline.run_pipeline()

#figures
#pipeline = AdvancedDemandForecastPipeline(df, output_dir=OUTPUT_DIR)
#pipeline.preprocess()
#pipeline.plot_raw_data()




