import json
import re
import random
import nltk
import multiprocessing
import os
import numpy as np
from multiprocessing import Pool, cpu_count
from pprint import pprint
from tqdm import tqdm
from string import punctuation
from itertools import chain
from nltk.corpus import stopwords


def vprint(msg, verbose=False):
    """
    Verbose print implementation.

    Arguments:
        msg(str): Message to 
        verbose(bool): 
    """
    if verbose:
        print(msg)
    else:
        pass

def vpprint(msg, verbose=False):
    """
    """
    if verbose:
        pprint(msg)
    else:
        pass

def load_rules(file):
    """
    .tsv files ONLY
    """
    assert file.endswith(".tsv")
    rules = list()
    with open(file, "r") as f:
        lines = f.read().splitlines()
        for itr in lines:
            rules.append(itr.split("\t"))
    #
    return rules

def readlines(file, begin=None, limit=None, rand=False, skip=False):
    """
    Read and split all content in the files line by line.

    Arguments:
        file(str): File to be read.
        begin(int): Index of the first line to be read.
        limit(int): Index of the last line to be read, or the amount of
            samples drawn from the dataset if rand is asserted.
        rand(bool): Randomly drawn samples from the data.
    Return:
        raw_data(list of strings): Lines from the files
    """
    print("Loading lines in the file...")
    if skip:
        with open(file, "r") as f:
            if rand:
                print(" - Random sample {:d} entries from data.".format(limit))
                data = f.read().splitlines()
                n_data = len(data)
                index = sorted(random.sample(list(range(n_data)), limit))
                data = [data[itr] for itr in index]
            else:
                data = f.read().splitlines()[1:limit]
                result = list()
                for abstract in data:
                    tab_index = abstract.find('\t')
                    abstract = abstract[tab_index+1:] # skip PMID\t
                    result.append(abstract)
                data = result
    else:
        with open(file, "r") as f:
            if rand:
                print(" - Random sample {:d} entries from data.".format(limit))
                data = f.read().splitlines()
                n_data = len(data)
                index = sorted(random.sample(list(range(n_data)), limit))
                data = [data[itr] for itr in index]
            else:
                data = f.read().splitlines()[begin:limit]
    print("Total {0} lines loaded.".format(len(data)))
    return data


def write_to_file(file, data):
    """
    Write strings to files.
    Saving choices available: JSON [.json], TSV [.tsv]

    Arguments:
        file(str): Output filename, carefully deal with the extension.
        data(list): List of list of strings OR Dictionary.
            
    """
    as_type = "JSON" if file.lower().endswith("json") else \
              "TSV"  if file.lower().endswith("tsv")  else \
              "TXT"  if file.lower().endswith("txt")  else \
              None
    #
    print("Writing result to file...")
    with open(file, "w") as f:
        if as_type == "JSON":
            # sort_keys = type(list(data.keys())[0]) == int or \
            #             type(list(data.keys())[0]) == float
            # json.dump(data, f, sort_keys=sort_keys, indent=4)
            json.dump(data, f, sort_keys=True, indent=4)
        elif as_type == "TSV" or as_type == "TXT":
            if type(data[0]) == list:
                for itr in tqdm(list(chain.from_iterable(data))):
                    f.write(itr + "\n")
            else:
                for itr in tqdm(data):
                    f.write(itr + "\n")
        else:
            print("[Type Error] Please specify type as JSON or TSV")
            exit()
    print("File saved in {:s}".format(file))

def split_data(data, n_slice, mode="TUPLE"):
    """
    Split data to minibatches with last batch may be larger or smaller.

    Arguments:
        data(ndarray): Array of data.
        n_slice(int): Number of slices to separate the data.
        mode(str): ["TUPLE", "SINGLE"]

    Return:
        partitioned_data(list): List of list containing any type of data.
    """
    n_data = len(data) if type(data) is list else data.shape[0]
    # Slice data for each thread
    print(" - Slicing data for threading...")
    print(" - Total number of data: {0}".format(n_data))
    per_slice = int(n_data / n_slice)
    partitioned_data = list()
    for itr in range(n_slice):
        # Generate indices for each slice
        idx_begin = itr * per_slice
        # last slice may be larger or smaller
        idx_end = (itr + 1) * per_slice if itr != n_slice - 1 else None
        #
        if mode == "TUPLE":
            partitioned_data.append((itr, data[idx_begin:idx_end]))
        elif mode == "SINGLE":
            partitioned_data.append(data[idx_begin:idx_end])
        else:
            print("Mode must be choice of [\"TUPLE\", \"SINGLE\"]")
            return None
    #
    return partitioned_data

def generic_threading(n_jobs, data, method, param=None, shared=False):
    """
    Generic threading method.

    Arguments:
        n_jobs(int): number of thead to run the target method
        data(ndarray): Data that will be split and distributed to threads.
        method(method object): Threading target method
        param(tuple): Tuple of additional parameters needed for the method.
        shared: (undefined)

    Return:
        result(list of any type): List of return values from the method.
    """
    # Threading settings
    n_cores = cpu_count()
    n_threads = n_cores * 2 if n_jobs == None else n_jobs
    print("Number of CPU cores: {:d}".format(n_cores))
    print("Number of Threading: {:d}".format(n_threads))
    #
    thread_data = split_data(data, n_threads)
    if param is not None:
        thread_data = [itr + param for itr in thread_data]
    else:
        pass
    #
    print(" - Begin threading...")
    # Threading
    with Pool(processes=n_threads) as p:
        if not shared:
            result = p.starmap(method, thread_data)
        else:
            p.starmap(method, thread_data)
    #
    print("\n" * n_threads)
    print("All threads completed.")
    return result if not shared else None

def multi_mentions(name):
    """

    Argument:
        name(str):

    Return:
        synonym(list of str): List of all synonyms.
    """
    synonym = None
    # A, B, or C
    if ", or " in name:
        parsed = name.replace(" or", "")
        synonym = parsed.split(", ")
    # A or B
    elif " or " in name:
        synonym = name.split(" or ")
    # A, B -> A, BA
    # A, B, C -> A, BA, CBA
    elif ", " in name:
        parsed = name.split(", ")
        synonym = [" ".join(reversed(parsed[: i + 1]))
                   for i in range(len(parsed))]
    # normal (no synonyms)
    else:
        synonym = [name]

    return synonym

def punctuation_cleanup(thread_idx, data, rules, mode):
    """
    Generic clean up the given corpus, the cleanup process can be either
    corpus-level (preliminary cleanup) or finer cleanups, i.e. vocabulary-
    level (for gaining purer vocabularies).

    Arguments:
        thread_idx(int): Indicating the thread ID, used for the positioning
                         and information of the progressbar.
        data(list of str): Each entry is a corpus to be processed.
        rules(list of tuples):
        mode(str): Mode of the cleanup method, available modes are
                SPLIT_WORDS: Used in extracting the vocabularies.
                PRELIMINARY: Preliminary rules for cleaning up the corpus.
    Return:
        linewords(list of str): List of either corpus (PRELIMINARY) or
                words (SPLIT_WORDS)
    """
    desc = "Thread {:2d}".format(thread_idx + 1)
    ########### EXCEPTION HANDLING ########### (TO-BE-IMPELMENTED)
    # assert mode 

    linewords = list()
    for article in tqdm(data, position=thread_idx, desc=desc):
    # for article in data:
        # replace tabs as spaces
        article = article.replace("\t", " ")
        # SPLIT_WORDS: used in finding vocabularies
        if mode == 'SPLIT_WORDS':
            # skip PMID
            vocabulary = article.translate(punctuation).lower().split()
            # cleanup some redundant punctuation
            for itr in rules:
                pattern, _ = itr
                # symbols at the end
                if pattern.startswith("*"):
                    symbol = pattern[1:]
                    vocabulary = [i[:-len(symbol)] if i.endswith(symbol)
                                  and not "-" in i else i
                                  for i in vocabulary]
                # symbols in the beginning
                elif pattern.endswith("*"):
                    symbol = pattern[:-1]
                    vocabulary = [i[len(symbol):] if i.startswith(symbol)
                                  and not "-" in i else i
                                  for i in vocabulary]
                else:
                    vocabulary = [i.replace(pattern, "") for i in vocabulary]

            linewords.append(vocabulary)
        # PRELIMINARY:
        elif mode == 'PRELIMINARY':
            for itr in rules:
                pattern, tag = itr
                found = re.findall(pattern, article)
                if len(found) == 0:
                    continue
                # support * replacement
                if tag.startswith("*"):
                    tag = tag[1:]
                    for itr_found in found:
                        article = article.replace(itr_found, itr_found.replace(tag, tag + " "))
                else:
                    for itr_found in found:
                        article = article.replace(itr_found, tag)
            linewords.append(article)
        else:
            print("Invalid mode type: {0}".format(mode))

    if mode == "SPLIT_WORDS":
        linewords = list(chain.from_iterable(linewords))
    return linewords

def corpus_cleanup(thread_idx, data, parentheses, refine_list):
    """
    Used to clean up the corpus according to the predefined regex commands.
    This method can be used standalone or passed to threads to improve
    performance.

    Arguments:
        thread_idx(int): Indicating the thread ID, used for the positioning
                         and information of the progressbar.
        data(list of str): Each entry is a corpus to be processed.
        parentheses(list of tuples): List containing patterns in the
            parentheses that are to be replaced by tags.
        refine_list(list of tuples):List containig patterns to be replaced
            by tags.
    Returns:
        result(list of str): Processed version of the input "data"
    """
    desc = "Thread {:2d}".format(thread_idx + 1)
    #
    result = list()
    for article in tqdm(data, position=thread_idx, desc=desc):
        article = article[article.find("\t") + 1:] # skip PMID\t
        # refine the corpus
        # Find contents within parentheses 
        contents = re.findall(r"\(.*?\)", article)
        for itr_content in contents:
            for itr_tag in parentheses:
                # extract entry
                pattern, tag = itr_tag
                # find pattern
                found = re.findall(pattern, itr_content)
                if len(found) != 0:
                    # add redundant spaces to avoid words stay together
                    article = article.replace(itr_content, " " + tag + " ")
                else:
                    pass
        # Find and replace patterns in the article
        for itr_pattern in refine_list:
            pattern, tag = itr_pattern
            article = re.sub(pattern, " " + tag + " ", article)
        #
        result.append(article.lower())

    return result

def keyword_in_sentences(thread_idx, data, keywords, mode="SINGLE"):
    """
    Sentence keyword search (called by threads or used normally)

    Arguments:
        thread_idx(int): Order of threads, used to align progressbars.
        data(list of str): Each elements in the list contains one sentence
                          of raw corpus.
        keywords(list of str): Contains all the keywords.
        [TO-BE-IMPLEMENTED] mode(str): MULTI or SINGLE

    Returns:
        result(list of str): Each elements in the list contains one 
                     sentence with one or more keywords.
    """
    # print("Marking the sentence in {:s} mode.".format(mode))
    desc = "Thread {:2d}".format(thread_idx + 1)
    result = list()
    found, found_sentence = None, None
    #
    for line in tqdm(data, position=thread_idx, desc=desc):
        # split sentences into word-level
        words = nltk.word_tokenize(line)

        ### TO-BE-REVISED ###
        # Conduct preliminary partial matching for keywords
        # Create a subset of keywords that partially matched.
        found_keyword = list()
        found_sentence = False
        partial_matched = list()
        for itr in keywords:
            # Append candidate to the list
            if itr.lower() in line.lower():
                partial_matched.append(itr)

        # Complete keyword matching on partial matched keywords
        # Length of the sentence of interest
        len_sentence = len(words)
        # Use sliding windows according to the keyword size
        for itr in partial_matched:
            found_word = None
            # Sliding window size = number of words in the keyword
            len_window = len(itr.split())
            # Slide the window through the entire sentence
            for begin in range(len_sentence - len_window):
                # Reconstruct the sentence for matching keywords
                tmp = " ".join(words[begin:begin + len_window])
                # Matched if the window contains the keyword
                if tmp.lower() == itr.lower():
                    found_word, found_sentence = True, True
                    found_keyword.append(itr)
                    # Next keyword
                    break
                else:
                    pass
            if mode == "SINGLE" and found_word:
                break

        # Remove duplicate matching
        if found_sentence:
            found_keyword = list(np.unique(found_keyword))
            result.append(line + "\t" + "\t".join(found_keyword))
    return result

def keywords_as_labels(thread_idx, data, keywords, labels, subwords=None,
                       mode=None, duplicate=True):
    """
    Arguments:
        thread_idx(int): Order of threads, used to align progressbars.
        data(list): List of data.
        keywords(dict): Dictionary that contains all mentions in interest.
        labels(dict): Dictionary that contains the mapping of types to int.
        subwords(dict): Dictionary that contains the subwords of all mentions in interest.
        mode(str): Choice ["SINGLE", "MULTI"].
        duplicate(bool): Whether to duplicate the sentence if it contains
                         multiple mentions.
    
    Returns:
        result(list of str): List of strings that are to be written to files.
    """
    desc = "Thread {:2d}".format(thread_idx + 1)
    result = list()
    #
    for line in tqdm(data, position=thread_idx, desc=desc):
        # Find split point and partition the string
        split_point = line.find("\t")
        sentence = line[:split_point]
        mentions = line[split_point + 1:].split("\t")

        # Add types (labels) of mentions to the data
        # [LABELS] \t [SENTENCE] \t [MENTION + SUBWORDS]
        # Single-Mention Mode (SMM)
        if mode == "SINGLE":
            # Fetch the first found mention
            entity_types = keywords[mentions[0]]

            # Retrived the types of the mentions
            # replace = ["__label__" + str(labels[itr]) for itr in entity_types]
            replace = [str(labels[itr]) for itr in entity_types]
            # SINGLE MENTION
            content = [",".join(replace), sentence, mentions[0]]
            result.append("\t".join(content))
            # FastText classification form
            # result.append( " , ".join(replace) + " " + sentence)

        # Multi-Mention Mode (MMM)
        elif mode == "MULTI":
            # Duplicate the sentence if the sentence contains more than one mentions
            if duplicate:
                # Fetch all mentions
                for itr_mention in mentions:
                    # Lookup the types of the mention in interest
                    entity_types = keywords[itr_mention]

                    # Convert entity types to labels
                    replace = [str(labels[itr]) for itr in entity_types]

                    # Concatenate subwords information with the mentions
                    if subwords is not None:
                        # Find the longest synonym given a set of mentions
                        long_mention = multi_mentions(itr_mention)[-1]
                        # Tokenize + lower case the mentions with more than one words
                        words = nltk.word_tokenize(long_mention.lower())
                        # tmp = words.copy()

                        # Remove punctuations which SHOULD BE CONSISTENT with that used in subword model
                        # ***TO-BE-IMPLEMENTED AS LOADING RULES FROM EXTERNAL FILE***
                        words = [re.sub("[" + punctuation + "0-9]", "", itr) for itr in words]

                        # Acquire all subwords for each word in mentions
                        subword_list = list()
                        for itr in words:
                            # Retrive subword information from given dictionary
                            try:
                                subword_list.append(subwords[itr])
                            # Skip words that do not exist in the subword dictionary
                            # Cause:
                            #   1. words with unfiltered or mis-filtered punctuation
                            #   2. empty string (produced by filtering figures)
                            #   3. TO-BE-DISCOVERED
                            except:
                                # Debugging
                                # print("{0} | {1} | *{2}*".format(tmp, words, itr))
                                # print("*{0}*".format(itr))
                                pass

                        # Unpack the subword list [list of subwords for each mention]
                        subword_list = list(chain.from_iterable(subword_list))

                        # remove stopword
                        stop_words = set(stopwords.words('english'))
                        filtered_subword_list = [w for w in subword_list if not w in stop_words]
                        # Sort all subwords in decending length order
                        # ***The filtering criteria should not be heuristic***
                        filtered_subword_list = sorted(filtered_subword_list, key=len, reverse=True)

                        # Mention Features: [mention _ subwords]
                        mention_features = itr_mention + " " + " ".join(filtered_subword_list[1:] if len(words) == 1 else filtered_subword_list)

                    # If no subword information is given, then the feature is just the mentions
                    else:
                        mention_features = itr_mention

                    # Add content to list
                    # [LABELS] \t [SENTENCE] \t [MENTION + SUBWORDS]
                    content = [",".join(replace), sentence, mention_features]
                    result.append("\t".join(content))

            # ***NOT DUPLICATING THE SENTENCE IF THERE ARE MORE THAN ONE MENTIONS***
            # Multi-Mention not duplicating (TO-BE-IMPLEMENTED)
            else:
                pass
            # replace = [str(labels[itr]) for itr in mentions]

        # Undefined mode
        else:
            pass

    return result

def merge_dict(path, trim=True, save_to_file=False, postfix=None):
    """
    Args:
        path(str): Path to where the dictionaries are saved.
        trim(bool): Use trimmed labels or not.
        save_to_file(bool): Save merged dictionaries to file if asserted.

    Returns:
        entity(dict): Merged keywords dictionaries.
    """
    # Parse directory name
    if not path.endswith("/"):
        path += "/"

    # File postfix
    label_type = postfix if postfix is not None else \
                 "_trimmed.json" if trim else "_leaf.json"

    print("Loading dictionaries from directory: {:s}".format(path))
    print(" - Use {:s} labels".format(postfix if postfix is not None else \
                                             "trimmed" if trim else "un-trimmed"))
    files = [itr for itr in os.listdir(path) if itr.endswith(label_type)]
    # Load and merge all dictionaries
    print("Found {:d} keywords dictionaries:".format(len(files)))
    for itr in files:
        print("- {:s}".format(itr))
    print()

    # Load and merge dictionaries
    entity = dict()
    for itr in files:
        entity.update(json.load(open(path + itr, "r")))
    print("{0} entities loaded from dictionaries".format(len(entity)))

    # Save merged dictionaries to file
    if save_to_file:
        file = "data/keywords.json"
        write_to_file(file, entity)

    print()
    return entity
