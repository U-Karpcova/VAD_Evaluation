from src.parsers.textgrid_parser import TextGrid_Parser
from src.metrics.vad_metrics import VAD_Metrics
from src.data.data_extractor import Data_Extractor
# from src.models.ten_vad_wrapper import TEN_VAD_MODEL
# from src.models.fire_red_wrapper import FireRed_VAD_MODEL
from src.models.silero_wrapper import Silero_Model

import pandas as pd
from pathlib import Path
from datetime import datetime

textgrid_parser = TextGrid_Parser()
evaluator = VAD_Metrics()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = BASE_DIR / "vad_results"
DATA_PATH = BASE_DIR / "data" / "sample_dataset"

def run_vad_pipeline(dataset_df, vad_model, textgrid_parser, evaluator):

  df = dataset_df.copy()
  # textgrid_parser. зчитування інформації з textgrid для кожної знайденої пари (wav, textgrid)
  df['pairs'] = df['pairs'].apply(lambda pairs_list: [(wav, textgrid_parser.build_vad_segments_detailed(tg)) for (wav, tg) in pairs_list])
  # vad model. prediction
  df['pairs'] = df['pairs'].apply(lambda pairs_list: [(vad_model.detect_segments(wav), tg) for (wav, tg) in pairs_list])

  # for each speaker
  metrics = {}
  for (name, pairs_list) in zip(df['name'], df['pairs']):
     metrics[name] = evaluator.compute_dataset_metrics(pairs_list)

  # for all dataset
  all_data = []
  for pairs_list in df['pairs']:
    all_data += pairs_list
  metrics['all_spk'] = evaluator.compute_dataset_metrics(all_data)

  metrics_df = pd.DataFrame(data=metrics)

  today = datetime.now().strftime("%d-%m-%Y")
  output_file = RESULTS_PATH / f"{vad_model.name}_{today}.json"
  metrics_df.to_json(output_file)

  return metrics_df

vad_models = [Silero_Model(),] # TEN_VAD_MODEL(), FireRed_VAD_MODEL()

extractor = Data_Extractor(DATA_PATH)
dataset_df, _ = extractor.collect_vad_dataset()
# excluded_ids=["Zb_Mozgh", ] # пошкоджені дані
# speaker_ids=['Zb_Karpt', 'R_Netre'] # тест

results = []
for vad_model in vad_models:
  metrics_df = run_vad_pipeline(dataset_df, vad_model, textgrid_parser, evaluator)
  results.append(metrics_df)

print(f'Results are saved to: {RESULTS_PATH}')