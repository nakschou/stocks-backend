from flask import Flask, request, json
import requests
import dspy
import os
from dotenv import load_dotenv
import json
from openai import OpenAI

app = Flask(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
turbo = dspy.OpenAI(model='gpt-4', api_key=OPENAI_API_KEY)
dspy.configure(lm=turbo)
client = OpenAI()

@app.route('/get_trending_stocks')
def get_trending_stocks():
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            },
            data=json.dumps({
                "model": "perplexity/sonar-small-chat", # Optional
                "messages": [
                {"role": "user", "content": "List 5 U.S stocks (and their tickers) trending today. They must be valid stocks."}
                ]
            })
        )
        text = response.json()["choices"][0]["message"]["content"]
        class DeriveStocks(dspy.Signature):
            """Given a message, derive the stocks and the tickers mentioned in the message.
            If the ticker is not mentioned, the model will try to predict the ticker based on the stock name."""

            message = dspy.InputField()
            stock1 = dspy.OutputField(desc="The first company name or N/A if not mentioned")
            ticker1 = dspy.OutputField(desc="The first company ticker or N/A if not mentioned")
            stock2 = dspy.OutputField(desc="The second company name or N/A if not mentioned")
            ticker2 = dspy.OutputField(desc="The second company ticker or N/A if not mentioned")
            stock3 = dspy.OutputField(desc="The third company name or N/A if not mentioned")
            ticker3 = dspy.OutputField(desc="The third company ticker or N/A if not mentioned")
            stock4 = dspy.OutputField(desc="The fourth company name or N/A if not mentioned")
            ticker4 = dspy.OutputField(desc="The fourth company ticker or N/A if not mentioned")
            stock5 = dspy.OutputField(desc="The fifth company name or N/A if not mentioned")
            ticker5 = dspy.OutputField(desc="The fifth company ticker or N/A if not mentioned")
        stock_derive = dspy.Predict(DeriveStocks)
        answer = stock_derive(message=text)
        response = app.response_class(
            response=json.dumps(answer.toDict()),
            status=200,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        response = app.response_class(
            response=json.dumps({"error": str(e)}),
            status=500,
            mimetype='application/json'
        )
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response