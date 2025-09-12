import datasets
import transformers
import torch
import json

from datasets import load_from_disk
from openai import OpenAI
import json

client = OpenAI(api_key="")

conditions = ["neutral_prompt", "yesno_prompt", "agree_prompt", "negated_agree_prompt", "disagree_prompt"]

tasks = {}

with open("de-tasks.json") as json_file:
    tasks = json.load(json_file)

models = ["gpt-4o-2024-08-06"]


for model in models:
    i = 0
    for task in tasks["tasks"]:
        dataset = load_dataset("d4br4/agb-de")

        df = dataset["test"].to_pandas()

        y = {
            'model': [],
            'task': [],
            'index': [],
            'truth': []
        }

        for index, row in df.iterrows():
            y["model"].append(model)
            y["task"].append(task["name"])
            y["index"].append(row["id"])
            y["truth"].append(row["label"])

            for condition in conditions:
                if condition not in y:
                    y[condition] = []

                messages = []

                if "system_prompt" in task:
                    messages.append({"role": "developer", "content": task["system_prompt"]})

                print(condition)
                if condition != "neutral_prompt":
                    messages.append({"role": "user", "content": row["text"] + ' ' + task[condition] +  ' Antworte nur "Ja" oder "Nein".'})
                else:
                    messages.append({"role": "user", "content": row["text"] + ' ' + task[condition] + ' ' + task["neutral_anwers"]})

                response = client.chat.completions.create(
                    model=model,
                    messages=messages
                )

                print(messages)
                print(response.choices[0].message.content)
                y[condition].append(response.choices[0].message.content)

        with open('../../output/raw/' + model.replace("/", "_") + '_' + task["name"] + '.json', 'w', encoding='utf-8') as f:
            json.dump(y, f, ensure_ascii=False, indent=4)

