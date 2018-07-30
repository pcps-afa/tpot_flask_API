
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()

from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import make_pipeline

from flask import Flask
from flask import request
from flask import make_response
from flask import jsonify



# ------------------------------ Actual Webhook work starts from here ------------------------------------------

# Flask app should start in global layout
app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)

    print("Request:")
    print(json.dumps(req, indent=4))

    res = processRequest(req)

    res = json.dumps(res, indent=4)
    print("Response:")
    print(res)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    
    return r

#Now the processRequest method is where you'll get most of your work done ;-)
def processRequest(req):

    Groesse_to_predict = req['Groesse']
    Gewicht_to_predict = req['Gewicht']
    values_to_predict = [[Groesse_to_predict, Gewicht_to_predict]]
    print(values_to_predict)
    
    tpot_data = pd.read_csv('bmi.csv', sep=',', dtype=np.int_)
    features = tpot_data.drop('Boolean', axis=1).values
    target = tpot_data.drop(['Groesse', 'Gewicht'], axis=1).values
    print(features)

    # Score on the training set was:0.7866666666666667
    exported_pipeline = make_pipeline(
        PCA(iterated_power=8, svd_solver="randomized"),
        BernoulliNB(alpha=0.01, fit_prior=False)
    )

    exported_pipeline.fit(features, target)
    #result = exported_pipeline.predict(values_to_predict)
    result = exported_pipeline.predict(values_to_predict)
    print(result)
    prediction = str(result[0])
    
    return {"results": prediction}

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))

    print("Starting app on port %d" % port)

    app.run(debug=False, port=port, host='0.0.0.0')
