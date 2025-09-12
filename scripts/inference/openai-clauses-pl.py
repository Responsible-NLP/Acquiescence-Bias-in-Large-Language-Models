import datasets
import transformers
import torch
import json

from datasets import load_dataset
from openai import OpenAI

conditions = ["neutral_prompt", "yesno_prompt", "agree_prompt", "negated_agree_prompt", "disagree_prompt"]

tasks = {}

with open("pl-tasks.json") as json_file:
    tasks = json.load(json_file)

client = OpenAI(api_key="")


models = ["gpt-4o-2024-08-06"]


for model in models:

    i = 0
    for task in tasks["tasks"]:
        dataset = load_dataset("laugustyniak/abusive-clauses-pl")

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
            y["index"].append(index)
            y["truth"].append(row["label"])

            for condition in conditions:
                if condition not in y:
                    y[condition] = []

                messages = []

                if "system_prompt" in task:
                    messages.append({"role": "system", "content": task["system_prompt"]})

                if condition != "neutral_prompt":
                    messages.append({"role": "user", "content": row["text"] + ' ' + task[condition] +  ' Po prostu odpowiedz "Tak" lub "Nie".'})
                else:
                    messages.append({"role": "user", "content": row["text"] + ' ' + task[condition] + ' ' + task["neutral_anwers"]})
                print(messages)

                response = client.chat.completions.create(
                    model=model,
                    messages=messages
                )

                print(response.choices[0].message.content)
                y[condition].append(response.choices[0].message.content)

        with open('../../output/raw/' + model.replace("/", "_") + '_' + task["name"] + '.json', 'w', encoding='utf-8') as f:
            json.dump(y, f, ensure_ascii=False, indent=4)

