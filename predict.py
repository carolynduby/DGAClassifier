import argparse
import sys
from src.model import QueryModel
import logging
from src.logging import setup_logging
import tldextract

import pickle

model_path = "trained.model"
#logging.info(f"Loading trained model from: {model_path}")
query_model = QueryModel(model_path, "legit")

def predict(args):
    data = dict(args)
    domain_name = data['domain']
    if "." in domain_name:
        domain_parts = tldextract.extract(data['domain'])
        domain_name = domain_parts.domain
        
    print("original", args, sep="=")
    print("second level", domain_name, sep="=")
    
    try:
          model_result = query_model.predict(domain_name)
    except ValueError:
        model_result='unknown'
    
    return {
        'legit': model_result
        }
    
   