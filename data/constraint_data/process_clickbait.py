import pandas as pd


def handle_stop_clickbait():
    df = pd.read_csv("train1.csv")
    df.rename({"headline": "text", "clickbait": "label"}, axis=1, inplace=True)
    df.to_csv(f"clickbait/stop_clickbait.csv", index=False)


def handle_clickbait_news_detection():
    df = pd.read_csv("train2.csv")
    df.rename({"title": "text"}, axis=1, inplace=True)
    df["label"] = (df["label"] == "clickbait").astype(int)
    df.to_csv(f"clickbait/clickbait_news_detection.csv", index=False)


if __name__ == "__main__":
    handle_stop_clickbait()
    handle_clickbait_news_detection()
