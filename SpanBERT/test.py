import sys
import re
import time
import requests
import spacy
from googleapiclient.discovery import build
from collections import defaultdict
from bs4 import BeautifulSoup
from spanbert import SpanBERT
from spacy_help_functions import extract_relations


# Load spacy model
nlp = spacy.load("en_core_web_lg")

# Load pre-trained SpanBERT model
spanbert = SpanBERT("./pretrained_spanbert")

# DEBUGGING MODE
DEBUGGING = False

# required relations
required_relations = {1: "per:schools_attended",
                      2: "per:employee_of",
                      3: "per:cities_of_residence",
                      4: "org:top_members/employees"}

# required named entity types for each relation type
entities_of_interest = {1: ['PERSON', 'ORGANIZATION'],
                        2: ['PERSON', 'ORGANIZATION'],
                        3: ['PERSON', 'LOCATION', 'CITY', 'STATE_OR_PROVINCE', 'COUNTRY'],
                        4: ['ORGANIZATION', 'PERSON']}


def callGoogleAPI(query):
    """
     Call Google Search API using auth info,
     and build a service object to search for the query.
     Return the top-10 search results
    """
    service = build("customsearch", "v1",
                    developerKey=client_key)
    res = service.cse().list(
        q=query,
        cx=engine_key,
    ).execute()
    return res['items'][:10]

def filter_nonascii(s):
    for i in range(len(s)):
        try:
            s[i].encode("ascii").decode()
        except:
            #means it's non-ASCII
            s = s.replace(s[i], " ") #replacing it with a single space
    return s

def get_plain_text_from_url(url):
    """
    Retrieve the webpage (Skip it if cannot retrieve due to timeout, etc.)
    Extract the actual plain text from the webpage using BeautifulSoup
    """
    # Retrieve the corresponding webpage (Skip it if cannot retrieve due to timeout, etc.)
    try:
        response = requests.get(url, timeout=3)
    except Exception as e:
        print("Unable to fetch URL. Continuing.")
        return

    # Extract the actual plain text from the webpage using BeautifulSoup
    # Reference code:
    # https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text/24968429#24968429
    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        # filter unwanted elements
        removed = ['[document]', 'style', 'noscript',
                   'title', 'meta', 'head', 'script']
        for s in soup(removed):
            s.extract()
        # get text
        #text = soup.get_text(separator="\n")
        text = soup.get_text()
        # separate into lines
        lines = [line.strip() for line in text.splitlines()]
        # break multi-headlines into a line each
        chunks = [phrase.strip() for line in lines for phrase in line.split("  ")]
        # remove blank lines and ignore non-ASCII characters
        text = ' '.join([chunk.encode('ascii', 'ignore').decode() for chunk in chunks if chunk])
        return text
    except Exception as e:
        print(e)
        print("No content in this URL. Continuing.")
        return


def main():
    """
    The main loop to improve Google search results using the
    user-provided relevance feedback
    """

    # Receive and check user's input
    if len(sys.argv) != 7:
        print("Usage: python3 main.py <google api key> <google engine id> <r> <t> <q> <k>")
        return

    global client_key, engine_key, r, t, q, k

    client_key = sys.argv[1]  # auth info's client key (developerKey)
    engine_key = sys.argv[2]  # auth info's engine key (cx)
    r = int(sys.argv[3])      # relation to extract
    t = float(sys.argv[4])    # extraction confidence threshold
    q = sys.argv[5]           # seed query of a plausible tuple
    k = int(sys.argv[6])      # the number of tuples in the output

    if r not in [1, 2, 3, 4]:
        print("Relation should be an integer between 1 and 4!")
        return
    if t < 0 or t > 1:
        print("Threshold should be a real number between 0 and 1!")
        return
    if k <= 0:
        print("The number of tuples should be an integer greater than 0!")
        return

    # print parameters info
    print("\n\n____")
    print("Parameters:")
    print("Client key	= ", client_key)
    print("Engine key	= ", engine_key)
    print("Relation	= ", required_relations[r])
    print("Threshold	= ", t)
    print("Query		= ", q)
    print("# of Tuples	= ", k)
    print("Loading necessary libraries; This should take a minute or so ...)")

    # Begin the main loop
    iteration = 0
    # Step 1: Initialize X, the set of extracted tuples, as the empty set
    X = defaultdict(int)
    url = "https://en.wikipedia.org/wiki/Megan_Rapinoe"
    print("URL: {}".format(url))

    # a. Retrieve the corresponding webpage (Skip it if cannot retrieve due to timeout, etc.)
    # b. Extract the actual plain text from the webpage using BeautifulSoup
    plain_text = get_plain_text_from_url(url)

    # c. Truncate the resulting plain text to its first 20,000 characters if longer than these
    if len(plain_text) > 20000:
        print("\tTrimming webpage content from {} to 20000 characters".format(len(plain_text)))
        plain_text = plain_text[:20000]
    print("\tWebpage length (num characters): {}".format(len(plain_text)))

    #print(plain_text)
    # d. Use spaCy to split the text into sentences and extract named entities (e.g., PERSON, ORGANIZATION)
    #print("\tAnnotating the webpage using spacy...")
    doc = nlp(plain_text)
    #print(doc)

    # e. Use the sentences and named entity pairs as input to SpanBERT to predict the corresponding
    #    relations, and extract all instances of the relation specified by r
    # f. Identify the tuples that have an associated extraction confidence >= t and add them to X
    # Step 4: Remove exact duplicated from X: if X contains tuples that are identical to
    # each other, keep only the copy that has the highest extraction confidence
    extract_relations(X, required_relations[r], doc, spanbert, entities_of_interest[r], conf=t)


if __name__ == '__main__':
    main()
