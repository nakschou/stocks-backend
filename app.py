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

@app.route('/get_trending_stocks', methods=['GET'])
def get_trending_stocks():
    def get_dct():
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            },
            data=json.dumps({
                "model": "perplexity/sonar-small-chat", # Optional
                "messages": [
                {"role": "user", "content": "List 5 U.S stocks (both company name and their tickers) outside of the top 20 that are trending today. They must be valid stocks."}
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
        dct = answer.toDict()
        return dct
    try:
        retries = 0
        cond = False
        while retries < 5 and not cond:
            dct = get_dct()
            if "N/A" in dct.values():
                retries += 1
            else:
                cond = True
        response = app.response_class(
            response=json.dumps(dct),
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
    
@app.route('/get_stock_ticker', methods=['GET'])
def get_stock_ticker():
    query = request.args.get('query', 'None')
    class DeriveTicker(dspy.Signature):
        """Given a search query containing a company name or ticker, derive the proper stock market ticker for the intended company."""

        search_query = dspy.InputField()
        ticker = dspy.OutputField(desc="The company ticker in one word, or N/A if the company is not found")
    try:
        stock_derive = dspy.Predict(DeriveTicker)
        answer = stock_derive(search_query=query)
        dct = answer.toDict()
        if dct["ticker"] == "N/A":
            response = app.response_class(
                response=json.dumps({"error": "No stock found for the given query"}),
                status=404,
                mimetype='application/json'
            )
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        response = app.response_class(
            response=json.dumps(dct),
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
