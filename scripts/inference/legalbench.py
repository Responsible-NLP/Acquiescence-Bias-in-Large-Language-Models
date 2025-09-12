import datasets
import transformers
import torch
import json

conditions = ["neutral_prompt", "yesno_prompt", "agree_prompt", "negated_agree_prompt", "disagree_prompt"]

tasks = {}

with open("tasks.json") as json_file:
    tasks = json.load(json_file)

models = ["meta-llama/Llama-3.1-8B-Instruct", "mistralai/Mistral-Small-24B-Instruct-2501", "google/gemma-2-27b-it", "meta-llama/meta-llama_Llama-3.3-70B-Instruct"]

for model in models:
    pipeline = transformers.pipeline(
        "text-generation", model=model, token="", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
    )

    i = 0
    for task in tasks["tasks"]:
        try:
            dataset = datasets.load_dataset("nguha/legalbench", task["name"], trust_remote_code=True)
        except:
            dataset = datasets.load_dataset("nguha/legalbench", task["name"])

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
            y["index"].append(row["index"])
            y["truth"].append(row["answer"])

            for condition in conditions:
                if condition not in y:
                    y[condition] = []

                messages = []
                system_instruction = ""
                if "system_prompt" in task:
                    if not model == "google/gemma-2-27b-it":
                        messages.append({"role": "system", "content": task["system_prompt"]})
                    else:
                        system_instruction = task["system_prompt"] + ' '

                print(condition)
                if condition != "neutral_prompt":
                    messages.append({"role": "user", "content": system_instruction + row["text"] + ' ' + task[condition] +  ' Just answer "Yes" or "No".'})
                else:
                    messages.append({"role": "user", "content": system_instruction + row["text"] + ' ' + task[condition] + ' ' + task["neutral_anwers"]})

                outputs = pipeline(messages, max_new_tokens=1000,)

                y[condition].append(outputs[0]['generated_text'][-1]['content'])

        with open('../../output/raw/' + model.replace("/", "_") + '_' + task["name"] + '.json', 'w', encoding='utf-8') as f:
            json.dump(y, f, ensure_ascii=False, indent=4)

