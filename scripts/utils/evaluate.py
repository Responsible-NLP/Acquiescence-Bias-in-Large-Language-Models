import pandas as pd
import os
from sklearn import metrics
from mlxtend.evaluate import mcnemar, mcnemar_table
import numpy as np

data = {
    "model": [],
    "task": [],
    "condition": [],
    "positive": [],
    "positive_change": [],
    "tp": [],
    "tp_change": [],
    "fn": [],
    "fn_change": [],
    "fp": [],
    "fp_change": [],
    "tn": [],
    "tn_change": [],
    "negative": [],
    "negative_change": [],
    "accuracy": [],
    "accuracy_change": [],
    "precision": [],
    "recall": [],
    "f1": [],
    "chi": [],
    "pv": [],
    "mc00": [],
    "mc01": [],
    "mc10": [],
    "mc11": []
}

def evaluateCsv(path, file):
    df = pd.read_csv(path + file)
    #df.sort_values(["model", "task", "condition"])

    for column in df:
        if column not in ["model", "task", "index", "truth"]:
            if df["task"][0] == "cuad_anti-assignment":
                continue

            modelname = df["model"][0]
            if "/" in modelname:
                modelname = modelname.split("/")[1].split("-Instruct")[0]

            data["model"].append(modelname)
            data["task"].append(df["task"][0])
            data["condition"].append(column)

            data["accuracy"].append(round(metrics.accuracy_score(df["truth"], df[column]),2))
            data["precision"].append(metrics.precision_score(df["truth"], df[column]))
            data["recall"].append(metrics.recall_score(df["truth"], df[column]))
            data["f1"].append(metrics.f1_score(df["truth"], df[column]))

            confusion_matrix = metrics.confusion_matrix(df["truth"], df[column])
            tn, fp, fn, tp = confusion_matrix.ravel()
            data["tn"].append(tn)
            data["fp"].append(fp)
            data["fn"].append(fn)
            data["tp"].append(tp)

            if column == "negated_agree_prompt":
                data["positive"].append(df[column].value_counts()[0])
                data["negative"].append(df[column].value_counts()[1])
            else:
                data["positive"].append(df[column].value_counts()[1])
                try:
                    data["negative"].append(df[column].value_counts()[0])
                except:
                    data["negative"].append(0)
            #data["positive"].append(tp + fp)
            #data["negative"].append(tn + fn)

            # calculate changes
            cm_neutral = metrics.confusion_matrix(df["truth"], df["neutral_prompt"])
            neutral_tn, neutral_fp, neutral_fn, neutral_tp = cm_neutral.ravel()

            #data["tp_change"].append(((tp - neutral_tp) / neutral_tp) * 100)
            #data["fp_change"].append(((fp - neutral_fp) / neutral_fp) * 100)
            #data["tn_change"].append(((tn - neutral_tn) / neutral_tn) * 100)
            #data["fn_change"].append(((fn - neutral_fn) / neutral_fn) * 100)

            data["tp_change"].append((tp - neutral_tp))
            data["fp_change"].append((fp - neutral_fp))
            data["tn_change"].append((tn - neutral_tn))
            data["fn_change"].append((fn - neutral_fn))

            pos = tp + fp
            neutral_pos = neutral_tp + neutral_fp
            data["positive_change"].append(((pos - neutral_pos) / neutral_pos) * 100)

            neg = tn + fn
            neutral_neg = neutral_tn + neutral_fn
            with np.errstate(divide='raise'):
                try:
                    data["negative_change"].append(((neg - neutral_neg) / neutral_neg) * 100)
                except:
                    data["negative_change"].append("n.a.")


            accuracy = metrics.accuracy_score(df["truth"], df[column])
            neutral_accuracy = metrics.accuracy_score(df["truth"], df["neutral_prompt"])
            data["accuracy_change"].append(((accuracy - neutral_accuracy) / neutral_accuracy) * 100)

            #stat sig
            if column == "neutral_prompt":
                data["chi"].append("")
                data["pv"].append("")
                data["mc00"].append("")
                data["mc01"].append("")
                data["mc10"].append("")
                data["mc11"].append("")
            else:
                tb = mcnemar_table(y_target=df["truth"],
                                   y_model1=df[column],
                                   y_model2=df["neutral_prompt"])

                data["mc00"].append(tb[0][0])
                data["mc01"].append(tb[0][1])
                data["mc10"].append(tb[1][0])
                data["mc11"].append(tb[1][1])

                chi, pv = mcnemar(ary=tb)
                data["chi"].append(round(chi,2))
                data["pv"].append(round(pv,4))



def addSummary(df):

    models = df['model'].unique()
    conditions = df['condition'].unique()

    for model in models:
        neutral = {}

        for condition in conditions:

            selected = df.loc[(df['model'] == model) & (df['condition'] == condition)]

            sum = {
                "model": model,
                "task": "overall",
                "condition": condition,
                "positive": selected["positive"].sum(),
                "positive_change": [],
                "tp": selected["tp"].sum(),
                "tp_change": [],
                "fn": selected["fn"].sum(),
                "fp": selected["fp"].sum(),
                "fp_change": [],
                "tn": selected["tn"].sum(),
                "negative": selected["negative"].sum(),
                "negative_change": [],
                "accuracy": round((selected["tp"].sum() + selected["tn"].sum()) / (selected["tp"].sum() + selected["fp"].sum() + selected["tn"].sum() + selected["fn"].sum()),2),
                "accuracy_change": [],
                "precision": selected["tp"].sum() / (selected["tp"].sum() + selected["fp"].sum()),
                "recall": selected["tp"].sum() / (selected["tp"].sum() + selected["fn"].sum()),
                "f1": (2 * + selected["tp"].sum())/(2*+ selected["tp"].sum() + + selected["fp"].sum() + + selected["fn"].sum()),
                "chi": [],
                "pv": [],
                "mc00": [],
                "mc01": [],
                "mc10": [],
                "mc11": []
            }

            if condition == "neutral_prompt":
                neutral = sum
                print("saved")
            else:
                sum["positive_change"] = ((sum["positive"] - neutral["positive"]) / neutral["positive"]) * 100
                sum["tp_change"] = ((sum["tp"] - neutral["tp"]) / neutral["tp"]) * 100
                sum["fp_change"] = ((sum["fp"] - neutral["fp"]) / neutral["fp"]) * 100
                sum["negative_change"] = ((sum["negative"] - neutral["negative"]) / neutral["negative"]) * 100
                sum["accuracy_change"] = ((sum["accuracy"] - neutral["accuracy"]) / neutral["accuracy"]) * 100

                #stat sig
                tb = np.array([[selected["mc00"].sum(), selected["mc01"].sum()], [selected["mc10"].sum(), selected["mc11"].sum()]])
                chi2, p = mcnemar(ary=tb)
                print(p)
                sum["chi"] = round(chi2, 2)
                sum["pv"] = round(p, 4)

            df = pd.concat([df, pd.DataFrame.from_records([sum])])

        print("--------")

    return df

def evaluateAll():
    path = "../../output/converted/"
    directory = os.fsencode(path)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".csv"):
            evaluateCsv(path, str(filename))

    pd.set_option('display.max_columns', None)
    df = pd.DataFrame(data)
    #print(df)
    df = addSummary(df)
    df.to_csv("../../output/evaluation.csv")

evaluateAll()