import json
import os
import pandas as pd
tasks = {}

de = {'total_ab': 0, 'total_yesno': 0, 'short_ab': 0, 'long_ab': 0, 'wrong_ab': 0, 'short_yesno': 0, 'long_yesno': 0, 'wrong_yesno': 0}
en = {'total_ab': 0, 'total_yesno': 0, 'short_ab': 0, 'long_ab': 0, 'wrong_ab': 0, 'short_yesno': 0, 'long_yesno': 0, 'wrong_yesno': 0}
pl = {'total_ab': 0, 'total_yesno': 0, 'short_ab': 0, 'long_ab': 0, 'wrong_ab': 0, 'short_yesno': 0, 'long_yesno': 0, 'wrong_yesno': 0}

def contains_any(s, substrings):
    return any(sub in s for sub in substrings)

def yesNoConv(yes, string, flip = False):
    if isinstance(string, int):
        return string

    if yes in ["Yes", "Ja", "Tak"]:
        de['total_yesno'] += 1

        if contains_any(string[:20].lower(), ["yes", "ja", "tak"]) or contains_any(string[:20].lower(),["no", "nein", "nie"]):
            if len(string) <= 5:
                de['short_yesno'] += 1
            else:
                de['long_yesno'] += 1
        else:
            de['wrong_yesno'] += 1
    else:
        de['total_ab'] += 1

        if contains_any(string[:20].lower(), yes):
            if len(string) + 1 <= len(yes) or len(string) <= 15:
                de['short_ab'] += 1
            else:
                de['long_ab'] += 1
        elif len(string) < 20:
            print(string)
        else:
            de['wrong_ab'] += 1



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


def convertAll():
    path = "../../output/raw/"
    directory = os.fsencode(path)

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".json"):
            convertJson(path, str(filename))



with open("answers.json", encoding="utf-8") as json_file:
    tasks = json.load(json_file)["tasks"]

convertAll()