groq_api_key = ""

open_source_models = ["llama2-70b-4096", "gemma-7b-it", "mixtral-8x7b-32768"]

risk_classes = ["explainable", "trustworthy"]

positive_definition_prompt = [f"""Provide a concise definition of highly {risk_classes[i]} AI inventory as viewed by public and Department of Homeland Security.
Definition of highly {risk_classes[i]} AI inventory within 100 words in a single paragraph:""" for i in range(len(risk_classes))]

positive_concept_prompt = [f"""Based on the context of a use case of an AI inventory provide a scenario when that use case would be most {risk_classes[i]} in the view of public and Department of Homeland Security.

Use Case:""" for i in range(len(risk_classes))]

negative_concept_prompt = [f"""Based on the context of a use case of an AI inventory provide a scenario when that use case would be least {risk_classes[i]} in the view of public and Department of Homeland Security.

Use Case:""" for i in range(len(risk_classes))]

trailing_answer = "\nConcise answer within 50 words:"




##### NLI Roberta Base ####

import requests

API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/nli-roberta-base-v2"
headers = {"Authorization": ""}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
# output = query({
# 	"inputs": {
# 		"source_sentence": "That is a happy person",
# 		"sentences": [
# 			"That is a happy dog",
# 			"That is a very happy person",
# 			"Today is a sunny day"
# 		]
# 	},
# })

# incontext = f"""Based on the context of a use case of an AI inventory provide a scenario when that use case would be most trustworthy in the view of public and Department of Homeland Security.

# Use Case: Generative Adversarial Networks (GANs)
# Answer:

# Use Case:
# Answer:

# Use Case:
# Answer:

# Use Case:
# Answer:
# """