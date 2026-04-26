# VAD_Evaluation
VAD Models Evaluation and Data Analysis project

Python 3.10

## Run pipeline
pip install torch==2.2.2+cpu torchaudio==2.2.2+cpu --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m src.pipeline.run_pipeline

## Run VAD Results UI
python -m vad_app.vad_gradio_app

## Run Dataset Analysis
pip install -r requirements.txt
python -m src.dataset_analysis.analysis_script

## Run Dataset Analysis UI
python -m analysis_app.dataset_analysis_app