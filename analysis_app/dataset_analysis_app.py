import gradio as gr
from src.parsers.stats_parser import DatasetStatsParser
from src.parsers.stats_parser import load_data
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE_DIR / "data_analysis_results"
dataset_df = load_data(RESULTS_PATH)


parser = DatasetStatsParser(dataset_df)

speakers = ["all"] + sorted(dataset_df["name"].unique().tolist())
emotions = list(parser.emotion_colors.keys())

with gr.Blocks() as demo:
        with gr.Tabs():

            # -------- TAB 1 --------
            with gr.Tab("Емоції"):

                with gr.Row():
                    with gr.Column(scale=1):
                        speaker_dd = gr.Dropdown(
                            choices=speakers,
                            value="all",
                            label="Спікер"
                        )

                        mode_dd = gr.Radio(
                            choices=["count", "time"],
                            value="count",
                            label="Метод обрахунку"
                        )

                    with gr.Column(scale=2):
                        plot_emotions = gr.Plot()

                def update_emotions(speaker, mode):
                    return parser.plot_emotions(speaker, mode)

                speaker_dd.change(update_emotions, [speaker_dd, mode_dd], plot_emotions)
                mode_dd.change(update_emotions, [speaker_dd, mode_dd], plot_emotions)

            # -------- TAB 2 --------
            with gr.Tab("Мовці"):

                with gr.Row():
                    with gr.Column(scale=1):
                        emotion_dd = gr.Dropdown(
                            choices=emotions,
                            value="anger",
                            label="Емоція"
                        )

                        mode_dd_2 = gr.Radio(
                            choices=["count", "time"],
                            value="count",
                            label="Метод обрахунку"
                        )

                    with gr.Column(scale=2):
                        plot_speakers = gr.Plot()

                def update_speakers(emotion, mode):
                    return parser.plot_speakers(emotion, mode)

                emotion_dd.change(update_speakers, [emotion_dd, mode_dd_2], plot_speakers)
                mode_dd_2.change(update_speakers, [emotion_dd, mode_dd_2], plot_speakers)

            # -------- TAB 3 --------
            with gr.Tab("Метадані"):

                with gr.Row():
                  mode_dd_meta = gr.Radio(
                    choices=["count", "time"],
                    value="count",
                    label="Метод обрахунку"
                    )

                # ---- 1. регіони + вік ----
                with gr.Row():
                  region_plot = gr.Plot()
                  age_plot = gr.Plot()

                # ---- 2. стать ----
                with gr.Row():
                  gender_plot = gr.Plot()

                # ---- 3. емоції за статтю ----
                with gr.Row():
                  gender_emotions_plot = gr.Plot()

                def update_meta(mode):
                  return (
                    parser.plot_regions(mode),
                    parser.plot_age(mode),
                    parser.plot_gender(mode),
                    parser.plot_emotions_by_gender(mode),
                  )

                mode_dd_meta.change(
                    fn=update_meta,
                    inputs=mode_dd_meta,
                    outputs=[
                      region_plot,
                      age_plot,
                      gender_plot,
                      gender_emotions_plot
                      ]
                        )


                # initial load
                demo.load(
                    fn=lambda: (
                        parser.plot_emotions("all", "count"),
                        parser.plot_regions("count"),
                        parser.plot_age("count"),
                        parser.plot_gender("count"),
                        parser.plot_emotions_by_gender("count"),
                        ),
                    outputs=[plot_emotions, region_plot, age_plot, gender_plot, gender_emotions_plot]
                    )

demo.launch()



