import os
import json
import pandas as pd
import gradio as gr
import plotly.express as px
from pathlib import Path

# Завантаження даних

def load_metrics(folder_path):
    data = {}

    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            model_name = file.replace(".json", "")
            with open(os.path.join(folder_path, file), "r") as f:
                data[model_name] = json.load(f)

    return data


BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = BASE_DIR / "results"

metrics_data = load_metrics(RESULTS_PATH)

# Допоміжні функції

def get_all_speakers():
    speakers = set()
    for model in metrics_data.values():
        speakers.update(model.keys())
    return sorted(list(speakers))

def get_all_metrics():
    # беремо метрики з першої моделі
    first_model = next(iter(metrics_data.values()))
    first_speaker = next(iter(first_model.values()))
    return list(first_speaker.keys())


ALL_SPEAKERS = get_all_speakers()
ALL_METRICS = get_all_metrics()
ALL_MODELS = list(metrics_data.keys())

model_color_map = {"fire_red_vad": "#ef553b", "silero_vad": "#636efa", "ten_vad": "#00cc96"}

# Графік для моделей

def plot_models(metric, speaker):
    results = []

    for model_name, model_data in metrics_data.items():
        if speaker not in model_data:
            continue

        value = model_data[speaker].get(metric, None)
        if value is not None:
            results.append({
                "model": model_name,
                "value": round(value, 4)
            })

    df = pd.DataFrame(results)
    df = df.sort_values(by="value", ascending=False)

    fig = px.bar(
        df,
        x="model",
        y="value",
        color="model",
        color_discrete_map=model_color_map,
        title=f"{metric} ({speaker})",
    )

    return fig


# Графік для мовців

def plot_speakers(metric, model):
    model_data = metrics_data.get(model, {})

    results = []

    for speaker, values in model_data.items():
        if metric in values:
            results.append({
                "speaker": speaker,
                "value": round(values[metric], 4)
            })

    df = pd.DataFrame(results)

    # сортування
    df = df.sort_values(by="value", ascending=False)

    fig = px.bar(
        df,
        x="speaker",
        y="value",

        title=f"{metric} ({model})",
    )

    color = model_color_map[model]
    fig.update_traces(marker_color=color)

    return fig


# Інтерфейс

with gr.Blocks() as demo:

    gr.Markdown("# VAD Metrics Dashboard")

    with gr.Tabs():

        # Вкладка 1: моделі
        with gr.Tab("Моделі"):

            with gr.Row():

                with gr.Column(scale=1):
                    metric_dd = gr.Dropdown(
                        choices=ALL_METRICS,
                        value="f1",
                        label="Метрика"
                    )

                    speaker_dd = gr.Dropdown(
                        choices=ALL_SPEAKERS,
                        value="all_spk",
                        label="Мовець"
                    )

                with gr.Column(scale=3):
                    model_plot = gr.Plot()

            metric_dd.change(
                plot_models,
                inputs=[metric_dd, speaker_dd],
                outputs=model_plot
            )

            speaker_dd.change(
                plot_models,
                inputs=[metric_dd, speaker_dd],
                outputs=model_plot
            )

            # initial plot
            demo.load(
                plot_models,
                inputs=[metric_dd, speaker_dd],
                outputs=model_plot
            )

        # Вкладка 2: мовці

        with gr.Tab("Мовці"):

            with gr.Row():

                with gr.Column(scale=1):
                    metric_dd_2 = gr.Dropdown(
                        choices=ALL_METRICS,
                        value="f1",
                        label="Метрика"
                    )

                    model_dd = gr.Dropdown(
                        choices=ALL_MODELS,
                        value=ALL_MODELS[0],
                        label="Модель"
                    )

                with gr.Column(scale=3):
                    speaker_plot = gr.Plot()

            metric_dd_2.change(
                plot_speakers,
                inputs=[metric_dd_2, model_dd],
                outputs=speaker_plot
            )

            model_dd.change(
                plot_speakers,
                inputs=[metric_dd_2, model_dd],
                outputs=speaker_plot
            )

            # initial plot
            demo.load(
                plot_speakers,
                inputs=[metric_dd_2, model_dd],
                outputs=speaker_plot
            )


demo.launch()