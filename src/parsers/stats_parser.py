import os
import pandas as pd
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
import json

def load_data(file_path):
    for file in os.listdir(file_path):
        if file.endswith(".json"):
            with open(os.path.join(file_path, file), "r") as f:
                json_data = json.load(f)
    return pd.DataFrame(data=json_data)

class DatasetStatsParser():
    def __init__(self, dataset_df) -> None:
        self.dataset_df = dataset_df

        # фіксовані кольори емоцій
        self.emotion_colors = {
            "anger": "#ff4646",  # червоний
            "joy": "#FFF45A",  # жовтий
            "sadness": "#636EFA",  # синій
            "fear": "#AB63FA",  # фіолетовий
            "disgust": "#c6e60c",  #
            "positive surprise": "#5AFFA1",  #
            "negative surprise": "#19D3F3",  #
            "contempt": "#088BA1", #
            "admiration": "#ffad9a", # рожевий
            "neutral": "#B0B0B0",   # сірий
            "other": "#6f6f6f",    # темно-сірий
            }

        self.emotion_name = {
            "an": "anger",
            "j": "joy",
            "sad": "sadness",
            "f": "fear",
            "dis": "disgust",
            "psur": "positive surprise",
            "nsur": "negative surprise",
            "con": "contempt",
            "ad": "admiration",
            "n": "neutral",
            "o": "other",
            }

    def __extract_emotion_name(self, emotion):
        if emotion in ['n', 'o']:
            return self.emotion_name[emotion]
        else:
          emotion = ''.join([c for c in emotion if not c.isdigit()])
        return self.emotion_name.get(emotion)


    def compute_emotions(self, speaker="all", mode="count"):
        emotion_stats = defaultdict(float)

        for _, row in self.dataset_df.iterrows():
            if speaker != "all" and row["name"] != speaker:
                continue

            for wav_path, segments in row["pairs"]:
                for seg in segments:
                    emo = self.__extract_emotion_name(seg.get("emotion"))
                    if emo and emo in self.emotion_name.values():

                        if mode == "count":
                            emotion_stats[emo] += 1
                        else:
                            duration = seg["end"] - seg["start"]
                            emotion_stats[emo] += duration

        result_df = pd.DataFrame({
            "emotion": list(emotion_stats.keys()),
            "value": list(emotion_stats.values())
        })

        return result_df.sort_values("value", ascending=False)


    def plot_emotions(self, speaker="all", mode="count"):
        df = self.compute_emotions(speaker, mode)

        fig = px.pie(
            df,
            names="emotion",
            values="value",
            color="emotion",
            color_discrete_map=self.emotion_colors,
        )

        fig.update_layout(
            title=f"Emotion distribution ({mode})",
            showlegend=True
        )

        return fig


    def compute_speakers(self, emotion="anger", mode="count"):
        speaker_stats = defaultdict(float)

        for _, row in self.dataset_df.iterrows():
            speaker = row["name"]

            for wav_path, segments in row["pairs"]:
                for seg in segments:
                    emo = self.__extract_emotion_name(seg.get("emotion"))

                    if emo != emotion:
                        continue

                    if mode == "count":
                        speaker_stats[speaker] += 1
                    else:
                        duration = seg["end"] - seg["start"]
                        speaker_stats[speaker] += duration

        result_df = pd.DataFrame({
            "speaker": list(speaker_stats.keys()),
            "value": list(speaker_stats.values())
        })

        return result_df.sort_values("value", ascending=False)

    def plot_speakers(self, emotion="anger", mode="count"):
        df = self.compute_speakers(emotion, mode)

        color = self.emotion_colors.get(emotion, "#333333")

        fig = px.bar(
            df,
            x="speaker",
            y="value",
            text_auto=True
        )

        # примусово один колір для всіх
        fig.update_traces(marker_color=color)

        fig.update_layout(
            title=f"Speakers distribution for '{emotion}' ({mode})",
            xaxis_title="Speaker",
            yaxis_title="Value",
            showlegend=False
        )

        return fig


    def compute_regions(self, mode="count"):
        stats = defaultdict(float)

        for _, row in self.dataset_df.iterrows():
            meta = row.get("metadata")
            if meta == 'Incorrect Metadata Format' or "region" not in meta:
                continue

            region = meta["region"]

            for wav_path, segments in row["pairs"]:
                if mode == "count":
                    stats[region] += 1
                else:
                    if segments:
                        duration = segments[-1]["end"]
                        stats[region] += duration

        return pd.DataFrame({
            "region": list(stats.keys()),
            "value": list(stats.values())
        }).sort_values("value", ascending=False)


    def compute_age(self, mode="count"):
        stats = defaultdict(float)

        for _, row in self.dataset_df.iterrows():
            meta = row.get("metadata")
            if meta == 'Incorrect Metadata Format' or "age_approx" not in meta:
                continue

            age = meta["age_approx"]

            for wav_path, segments in row["pairs"]:
                if mode == "count":
                    stats[age] += 1
                else:
                    if segments:
                        duration = segments[-1]["end"]
                        stats[age] += duration

        return pd.DataFrame({
            "age": list(stats.keys()),
            "value": list(stats.values())
        }).sort_values("age")


    def compute_gender(self, mode="count"):
        stats = defaultdict(float)

        for _, row in self.dataset_df.iterrows():
            meta = row.get("metadata")
            if meta == 'Incorrect Metadata Format' or "sex" not in meta:
                continue

            sex = meta["sex"]

            for wav_path, segments in row["pairs"]:
                if mode == "count":
                    stats[sex] += 1
                else:
                    if segments:
                        duration = segments[-1]["end"]
                        stats[sex] += duration

        return pd.DataFrame({
            "sex": list(stats.keys()),
            "value": list(stats.values())
        })


    def compute_emotions_by_gender(self, mode="count"):
        stats = defaultdict(lambda: {"M": 0, "F": 0})

        for _, row in self.dataset_df.iterrows():
            meta = row.get("metadata")
            if meta == 'Incorrect Metadata Format' or "sex" not in meta:
                continue

            gender = meta["sex"]

            for _, segments in row["pairs"]:
                for seg in segments:
                    emo = self.__extract_emotion_name(seg.get("emotion"))
                    if not emo:
                        continue

                    if mode == "count":
                        stats[emo][gender] += 1
                    else:
                        stats[emo][gender] += seg["end"] - seg["start"]

        # перетворюємо в DataFrame
        data = []
        for emo, values in stats.items():
          data.append({
            "emotion": emo,
            "M": values["M"],
            "F": values["F"]
            })

        df = pd.DataFrame(data)
        # щоб порядок емоцій був стабільний
        return df.sort_values("emotion")


    def plot_regions(self, mode="count"):
        df = self.compute_regions(mode)

        return px.bar(df, x="region", y="value", text_auto=True,
                  title=f"Regions ({mode})")


    def plot_age(self, mode="count"):
        df = self.compute_age(mode)

        return px.bar(df, x="age", y="value", text_auto=True,
                  title=f"Age ({mode})")


    def plot_gender(self, mode="count"):
        df = self.compute_gender(mode)

        return px.pie(
        df,
        names="sex",
        values="value",
        color="sex",
        color_discrete_map={
            "M": "#4A90E2",  # синій
            "F": "#FF69B4"   # рожевий
        },
        title=f"Gender ({mode})"
    )


    def plot_emotions_by_gender(self, mode="count"):
        df = self.compute_emotions_by_gender(mode)

        fig = go.Figure()

        # чоловіки
        fig.add_bar(
            x=df["emotion"],
            y=df["M"],
            name="Male",
            marker_color="#4A90E2"  # синій
        )

        # жінки
        fig.add_bar(
            x=df["emotion"],
            y=df["F"],
            name="Female",
            marker_color="#FF69B4"  # рожевий
        )

        fig.update_layout(
            title=f"Emotions by gender ({mode})",
            xaxis_title="Emotion",
            yaxis_title="Value",
            barmode="group"
        )

        return fig

