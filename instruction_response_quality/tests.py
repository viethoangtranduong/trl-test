from handler import EndpointHandler
import pandas as pd

# init handler
response_model_handler = EndpointHandler()

# prepare sample payload
payload = {'inputs': {"instruction": "What are some ways to stay energized throughout the day?",
           "response": "Drink lots of coffee!"}}

# test the handler
pred=response_model_handler(payload)

print(pred)

payload = {'inputs': [{"instruction": "What are some ways to stay energized throughout the day?",
                       "response": "Drink lots of coffee!",
                       "dataset": ''},
                       {"instruction": "What are some ways to stay energized throughout the day?",
                       "response": "Drink lots of coffee!",
                       "dataset": 'dolly'},
                      {"instruction": "What are some ways to stay energized throughout the day?",
                       "response": "Drink lots of coffee!",
                       "dataset": 'open-assistant'},
                     {"instruction": "What are some ways to stay energized throughout the day?",
                       "response": "Drink lots of coffee!",
                       "dataset": 'helpful_instructions'}]}

# test the handler
pred=response_model_handler(payload)

print(pred)