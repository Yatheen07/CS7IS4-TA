import pandas as pd
import os

sentiment_ranking_df = pd.read_csv("datasets/ranking_all_emojis.csv")

emoticons= sentiment_ranking_df[sentiment_ranking_df['Unicode block']=="Emoticons"]

filename = "emoticonsRanked.csv"
outdir = './datasets'
if not os.path.exists(outdir):
    os.mkdir(outdir)
fullname = os.path.join(outdir, filename)
emoticons.to_csv(fullname, index=False)
print("Saving dataset.")