import logging, sys, argparse


def str2bool(v):
    # copy from StackOverflow
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_entity(tag_seq, char_seq):
    # PER = get_PER_entity(tag_seq, char_seq)
    # LOC = get_LOC_entity(tag_seq, char_seq)
    # ORG = get_ORG_entity(tag_seq, char_seq)
    SIGNS=get_SIGNS_entity(tag_seq, char_seq)
    BODY=get_BODY_entity(tag_seq, char_seq)
    CHECK=get_CHECK_entity(tag_seq, char_seq)
    TREATMENT=get_TREAMENT_entity(tag_seq, char_seq)
    DISEASE=get_DISEASE_entity(tag_seq,char_seq)

    #return PER, LOC, ORG
    return SIGNS,BODY,CHECK,TREATMENT,DISEASE

def get_CHECK_entity(tag_seq,char_seq):
    length = len(char_seq)
    CHECK = []
    for i ,(char,tag) in enumerate(zip(char_seq,tag_seq)):
        if tag == "B-CHECK":
            check = char
        elif tag == "I-CHECK":
            check += char
        elif tag == "E-CHECK":
            check +=char
            CHECK.append(check)
            del check
        elif tag == "S-CHECK":
            check =char
            CHECK.append(check)
        else:
            continue
    return CHECK

def get_SIGNS_entity(tag_seq, char_seq):
    length = len(char_seq)
    SIGNS = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == "B-SIGNS":
            signs = char
        elif tag == "I-SIGNS":
            signs += char
        elif tag == "E-SIGNS":
            signs += char
            SIGNS.append(signs)
            del signs
        elif tag == "S-SIGNS":
            signs = char
            SIGNS.append(signs)
            del signs
        else:
            continue
    return SIGNS

def get_TREAMENT_entity(tag_seq, char_seq):
    length = len(char_seq)
    TREAMENT = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == "B-TREAMENT":
            treament = char
        elif tag == "I-TREAMENT":
            treament += char
        elif tag == "E-TREAMENT":
            treament += char
            TREAMENT.append(treament)
            del treament
        elif tag == "S-TREAMENT":
            treament = char
            TREAMENT.append(treament)
            del treament
        else:
            continue
    return TREAMENT

def get_BODY_entity(tag_seq, char_seq):
    length = len(char_seq)
    BODY = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == "B-BODY":
            body = char
        elif tag == "I-BODY":
            body += char
        elif tag == "E-BODY":
            body += char
            BODY.append(body)
            del body
        elif tag == "S-BODY":
            body = char
            BODY.append(body)
            del body
        else:
            continue
    return BODY

def get_DISEASE_entity(tag_seq, char_seq):
    length = len(char_seq)
    DISEASE = []
    for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
        if tag == "B-DISEASE":
            disease = char
        elif tag == "I-DISEASE":
            disease += char
        elif tag == "E-DISEASE":
            disease += char
            DISEASE.append(disease)
            del disease
        elif tag == "S-DISEASE":
            disease = char
            DISEASE.append(disease)
            del disease
        else:
            continue
    return DISEASE



# def get_PER_entity(tag_seq, char_seq):
#     length = len(char_seq)
#     PER = []
#     for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
#         if tag == 'B-PER':
#             if 'per' in locals().keys():
#                 PER.append(per)
#                 del per
#             per = char
#             if i+1 == length:
#                 PER.append(per)
#         if tag == 'I-PER':
#             per += char
#             if i+1 == length:
#                 PER.append(per)
#         if tag not in ['I-PER', 'B-PER']:
#             if 'per' in locals().keys():
#                 PER.append(per)
#                 del per
#             continue
#     return PER
#
#
# def get_LOC_entity(tag_seq, char_seq):
#     length = len(char_seq)
#     LOC = []
#     for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
#         if tag == 'B-LOC':
#             if 'loc' in locals().keys():
#                 LOC.append(loc)
#                 del loc
#             loc = char
#             if i+1 == length:
#                 LOC.append(loc)
#         if tag == 'I-LOC':
#             loc += char
#             if i+1 == length:
#                 LOC.append(loc)
#         if tag not in ['I-LOC', 'B-LOC']:
#             if 'loc' in locals().keys():
#                 LOC.append(loc)
#                 del loc
#             continue
#     return LOC
#
#
# def get_ORG_entity(tag_seq, char_seq):
#     length = len(char_seq)
#     ORG = []
#     for i, (char, tag) in enumerate(zip(char_seq, tag_seq)):
#         if tag == 'B-ORG':
#             if 'org' in locals().keys():
#                 ORG.append(org)
#                 del org
#             org = char
#             if i+1 == length:
#                 ORG.append(org)
#         if tag == 'I-ORG':
#             org += char
#             if i+1 == length:
#                 ORG.append(org)
#         if tag not in ['I-ORG', 'B-ORG']:
#             if 'org' in locals().keys():
#                 ORG.append(org)
#                 del org
#             continue
#     return ORG


def get_logger(filename):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
