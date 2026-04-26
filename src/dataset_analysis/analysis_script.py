from src.parsers.textgrid_parser import TextGrid_Parser
from src.data.data_extractor import Data_Extractor
from pathlib import Path
from datetime import datetime

textgrid_parser = TextGrid_Parser()

# root_dir = "/content/drive/MyDrive/2026_Бакалаври_КНУ"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "sample_dataset"
RESULTS_PATH = BASE_DIR / "data_analysis_results"

extractor = Data_Extractor(DATA_PATH)
dataset_df, _ = extractor.collect_vad_dataset()
# excluded_ids=["Zb_Mozgh", ]
# speaker_ids=['Zb_Karpt', 'R_Fysyn']

dataset_df['pairs'] = dataset_df['pairs'].apply(lambda pairs_list: [textgrid_parser.extract_speech_segments(tg) for (wav, tg) in pairs_list])

today = datetime.now().strftime("%d-%m-%Y")
save_path = f'{RESULTS_PATH}/Exploratory_Data_Analysis_{today}.json'
dataset_df.to_json(save_path)

print(f'Dataset analysis completed. File saved to {save_path}')