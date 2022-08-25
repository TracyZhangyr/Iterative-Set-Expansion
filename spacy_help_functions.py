import spacy
from collections import defaultdict

spacy2bert = {
        "ORG": "ORGANIZATION",
        "PERSON": "PERSON",
        "GPE": "LOCATION",
        "LOC": "LOCATION",
        "DATE": "DATE"
        }

bert2spacy = {
        "ORGANIZATION": "ORG",
        "PERSON": "PERSON",
        "LOCATION": "LOC",
        "CITY": "GPE",
        "COUNTRY": "GPE",
        "STATE_OR_PROVINCE": "GPE",
        "DATE": "DATE"
        }


def get_entities(sentence, entities_of_interest):
    return [(e.text, spacy2bert[e.label_]) for e in sentence.ents if e.label_ in spacy2bert]


def extract_relations(X, required_relation, doc, spanbert, entities_of_interest, conf=0.7):
    """
    1. Extract named entities (e.g., PERSON, ORGANIZATION).
    2. Use the sentences and named entity pairs as input to SpanBERT to predict the corresponding
    relations, and extract all instances of the relation specified by r.
    3. Identify the tuples that have an associated extraction confidence >= t and add them to X.
    Remove exact duplicated from X: if X contains tuples that are identical to
    each other, keep only the copy that has the highest extraction confidence.
    """
    num_sentences = len([s for s in doc.sents])
    print("\tExtracted {} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...".format(num_sentences))
    processed_sentences_count = 0
    extracted_annotations_count = 0
    relations_extracted = 0
    overall_relations = 0

    # process each sentence in the doc
    for sentence in doc.sents:
        extract_annotations = False
        if processed_sentences_count > 0 and processed_sentences_count % 5 == 0:
            print("\tProcessed {} / {} sentences".format(processed_sentences_count, num_sentences))
        processed_sentences_count += 1

        # extract named entity pairs for the input to SpanBERT
        entity_pairs = create_entity_pairs(sentence, entities_of_interest)
        examples = []
        for ep in entity_pairs:
            # only add named entity pairs that contain named entities of the right type 
            # for the relation of interest r
            # e.g. r = 1 -> Schools_Attended: Subject: PERSON, Object: ORGANIZATION
            # entities_of_interest = ['PERSON', 'ORGANIZATION']
            # Subject = entities_of_interest[0], Object = entities_of_interest[1:]
            e1_type, e2_type = ep[1][1], ep[2][1]
            subject, objects = entities_of_interest[0], entities_of_interest[1:]

            if e1_type == subject and e2_type in objects:
                examples.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})
            elif e2_type == subject and e1_type in objects:
                examples.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})

        if len(examples) == 0:
            continue

        # use SpanBERT to predict the corresponding relations
        preds = spanbert.predict(examples)
        # extract all instances of the relation specified by r
        for ex, pred in list(zip(examples, preds)):
            relation = pred[0]
            if relation != required_relation:
                continue

            print("\n\t\t=== Extracted Relation ===")
            print("\t\tInput tokens: {}".format(ex['tokens']))

            extract_annotations = True
            overall_relations += 1
            subj = ex["subj"][0]
            obj = ex["obj"][0]
            confidence = pred[1]

            print("\t\tOutput Confidence: {:.7f} ; Subject: {} ; Object: {} ;".format(confidence, subj, obj))
            # identity the tuples that have an associated extraction confidence >= t and add them to X
            if confidence > conf:
                # Remove exact duplicated from X: if X contains tuples that are identical to
                # each other, keep only the copy that has the highest extraction confidence
                if X[(subj, obj)] < confidence:
                    relations_extracted += 1
                    X[(subj, obj)] = confidence
                    print("\t\tAdding to set of extracted relations")
                else:
                    print("\t\tDuplicate with lower confidence than existing record. Ignoring this.")
            else:
                print("\t\tConfidence is lower than threshold confidence. Ignoring this.")
            print("\t\t==========")
        
        
        if extract_annotations == True:
            extracted_annotations_count += 1
    
    print("Extracted annotations for  {}  out of total  {}  sentences".format(extracted_annotations_count, num_sentences))
    print("Relations extracted from this website: {} (Overall: {})".format(relations_extracted, overall_relations))
    return 


def create_entity_pairs(sents_doc, entities_of_interest, window_size=40):
    '''
    Input: a spaCy Sentence object and a list of entities of interest
    Output: list of extracted entity pairs: (text, entity1, entity2)
    '''
    if entities_of_interest is not None:
        entities_of_interest = {bert2spacy[b] for b in entities_of_interest}
    ents = sents_doc.ents # get entities for given sentence

    length_doc = len(sents_doc)
    entity_pairs = []
    for i in range(len(ents)):
        e1 = ents[i]
        if entities_of_interest is not None and e1.label_ not in entities_of_interest:
            continue

        for j in range(1, len(ents) - i):
            e2 = ents[i + j]
            if entities_of_interest is not None and e2.label_ not in entities_of_interest:
                continue
            if e1.text.lower() == e2.text.lower(): # make sure e1 != e2
                continue

            if (1 <= (e2.start - e1.end) <= window_size):

                punc_token = False
                start = e1.start - 1 - sents_doc.start
                if start > 0:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start -= 1
                        if start < 0:
                            break
                    left_r = start + 2 if start > 0 else 0
                else:
                    left_r = 0

                # Find end of sentence
                punc_token = False
                start = e2.end - sents_doc.start
                if start < length_doc:
                    while not punc_token:
                        punc_token = sents_doc[start].is_punct
                        start += 1
                        if start == length_doc:
                            break
                    right_r = start if start < length_doc else length_doc
                else:
                    right_r = length_doc

                if (right_r - left_r) > window_size: # sentence should not be longer than window_size
                    continue

                x = [token.text for token in sents_doc[left_r:right_r]]
                gap = sents_doc.start + left_r
                e1_info = (e1.text, spacy2bert[e1.label_], (e1.start - gap, e1.end - gap - 1))
                e2_info = (e2.text, spacy2bert[e2.label_], (e2.start - gap, e2.end - gap - 1))
                if e1.start == e1.end:
                    assert x[e1.start-gap] == e1.text, "{}, {}".format(e1_info, x)
                if e2.start == e2.end:
                    assert x[e2.start-gap] == e2.text, "{}, {}".format(e2_info, x)
                entity_pairs.append((x, e1_info, e2_info))
    return entity_pairs

