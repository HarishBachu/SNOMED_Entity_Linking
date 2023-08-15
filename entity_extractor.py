"""Extracts clinical entities from free text using LLMs."""
import fhir_api
from prompts import *
import json
import time
import re
import argparse

DEFAULT_MODEL = 'openai'

# Get system arguments (api, model)
arg_parser = argparse.ArgumentParser(prog='LLM CT Entity Extractor', description='Extracts entities from clinical text using LLMs.')
arg_parser.add_argument('-a', '--api', default=DEFAULT_MODEL, help='the API to use (options: llama, openai, bard)')
arg_parser.add_argument('--model', help='the model to run, dependent upon API choice')
_args = arg_parser.parse_args()
llm_api = _args.api.lower()

# Conditionally import the chat completion function which uses the given API
if llm_api == 'llama':
    from completion_llama import create_chat_completion
elif llm_api == 'openai':
    from completion_openai import create_chat_completion
elif llm_api == 'bard':
    from completion_bard import create_chat_completion
else:
    raise NotImplemented(f'Please set the api argument to one of (llama, openai, bard). Got {llm_api}.')

"""
def create_chat_completion_wrapped(prompt, **kwargs):
    # Override any model parameters here, for example, setting temperature to 0:
    kwargs['temperature'] = kwargs.get('temperature', 0)
    return create_chat_completion(prompt, **kwargs)
create_chat_completion = create_chat_completion_wrapped
"""

# Constants for the kind of match
DIRECT_MATCH, SIMPLIFIED_MATCH, GENERALISED_MATCH = range(3)

# ANSI escape sequences for text colors
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RESET = "\033[0m"

def colorize_text(text, replacements, color_code):
    for word in replacements:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        text = pattern.sub(f"{color_code}{word}{COLOR_RESET}", text)
    return text

def display_color(line, entities):
    print(line)
    print(entities)

server_url = "https://snowstorm.ihtsdotools.org/fhir"
# server_url = "http://localhost:8080"
valueset_url = "http://snomed.info/sct/900000000000207008/version/20230630?fhir_vs"
# valueset_url = "http://snomed.info/sct?fhir_vs=isa/138875005"

def match_snomed(term):
    # skip it term length is less than 3
    if len(term) < 3 or len(term) > 100:
        return None
    fhir_response = fhir_api.expand_valueset(server_url, valueset_url, term)
    best_match = None
    if (fhir_response and 'expansion' in fhir_response and 'contains' in fhir_response['expansion'] and len(fhir_response['expansion']['contains']) > 0):
        # Check if there is a case insensitive exact match in fhir_response['expansion']['contains']
        for item in fhir_response['expansion']['contains']:
            if item['display'].lower() == term.lower():
                best_match = item
                break
        # If there is no exact match, return the first match
        # print(fhir_response['expansion']['contains'])

        if not best_match:
            best_match = fhir_response['expansion']['contains'][0]
    return best_match

def rate(term, match, context):
    """Rate the accuracy of the assigned match to the term on a rating of 1 to 5, or 0 if the response is invalid."""
    if term.lower() == match.lower():
        return 5
    accuracy_prompts[-1]['content'] = "Clinician's term: {}\nSNOMED term: {}\nContext: {}".format(term, match, context)
    response = create_chat_completion(accuracy_prompts, max_tokens=2)
    if response in ('1', '2', '3', '4', '5'):
        return int(response)
    else:
        return 0

def from_prompt(prompts, term):
    prompts[-1]['content'] = term
    return prompts

def identify(text):
    """Return the clinical entities in a clinical note or sample of free text."""
    print(text)
    # Query the model for a chat completion that extracts entities from the text.
    json_text = create_chat_completion(from_prompt(extract_prompts, text))
    pattern = r'\[.*\]'
    # Search for a json array.
    match = re.search(pattern, json_text, re.DOTALL)
    if match:
        json_array = match.group()
        try:
            results = json.loads(json_array)
            response_terms = [result['text'] for result in results]
        except (json.decoder.JSONDecodeError, TypeError):
            print(COLOR_RED, "Invalid or malformed JSON:", json_array, COLOR_RESET)
            return {}
    else:
        return {}

    # Dictionary to assign each term a list of information about the match found, confidence and strategy used.
    term_results = {}

    for term in response_terms:
        term_results[term] = None

        # Look for direct matches with SNOMED CT
        potential_match = match_snomed(term)
        if potential_match:
            rating = rate(term, potential_match['display'], text)
            term_results[term] = (potential_match, rating, DIRECT_MATCH)
            if rating > 3:
                continue

        # Attempt to simplify the term in order to improve on the initial match or find a match
        simple_term = create_chat_completion(from_prompt(simplify_prompts, term))
        new_potential_match = match_snomed(term)
        if new_potential_match != potential_match:
            new_rating = rate(simple_term, new_potential_match['display'], term)
            if new_rating > rating:
                # Set the match or replace the previous match with the new match
                potential_match, rating = new_potential_match, new_rating
                term_results[term] = (potential_match, rating, SIMPLIFIED_MATCH, simple_term)
            if new_rating > 3:
                continue

        # Attempt to generalise the term
        general_term = create_chat_completion(from_prompt(generalise_prompts, term))
        new_potential_match = match_snomed(term)
        if new_potential_match != potential_match:
            new_rating = rate(general_term, new_potential_match['display'], term)
            if new_rating > rating:
                # Set the match or replace the previous match with the new match
                term_results[term] = (new_potential_match, new_rating, GENERALISED_MATCH, general_term)

    return term_results


def main():
    # Initialise the LLM we are using (if required)
    # Read the test cases (hide blank lines)
    with open("clinical_text.txt", "r") as file:
        lines = map(str.strip, file.readlines())
    
    entities_per_line = []

    # Iterate over each line in the test cases
    for line in lines:
        if not line or line.startswith('#'):  # skip newlines and comments/titles
            continue

        entities = identify(line)
        entities_per_line.append(entities)
        display_color(line, entities)


if __name__ == "__main__":
    main()
