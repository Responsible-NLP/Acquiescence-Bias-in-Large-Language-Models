import json
import os
import pandas as pd
tasks = {}

def yesNoConv(yes, string, flip = False):
    print(yes)
    print(string)
    if isinstance(string, int):
        return string

    truth = yes.lower() in string[:20].lower()

    if flip:
        truth = not truth

    if truth:
        return 1
    else:
        return 0

def convertJson(folder, filename):
    with open(folder + filename, encoding='utf-8') as inputfile:
        df = pd.read_json(inputfile)

        for column in df:
            if column not in ["model", "task", "index", "neutral_prompt"]:
                yes = "Yes"

                if "agb-de" in filename:
                    yes = "Ja"
                elif "clauses-pl" in filename:
                    yes = "Tak"

                df[column] = df[column].map(lambda x: yesNoConv(yes, x, flip=column == "disagree_prompt"))
            elif column == "neutral_prompt":
                task = {}
                for t in tasks:
                    if t["name"] == df["task"][0]:
                        task = t
                        break
                df[column] = df[column].map(lambda x: yesNoConv(task["neutral_yes"], x))

        df.to_csv(folder.replace("/raw/", "/converted/") + filename.replace(".json", ".csv"), encoding='utf-8', index=False)

def convertAll():
    path = "../../output/raw/"
    directory = os.fsencode(path)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if not "gemma" in filename:
            continue
        if filename.endswith(".json"):
            convertJson(path, str(filename))



with open("answers.json", encoding="utf-8") as json_file:
    tasks = json.load(json_file)["tasks"]

convertAll()