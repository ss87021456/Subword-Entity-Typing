import argparse
import json
import numpy as np
from itertools import chain
from collections import Counter
from utils import write_to_file, readlines, merge_dict

"""
python src/hierarchical_eval.py --labels=data/label.json \
--mention=mention_list.txt --prediction=... \
--k_parents=5 --hierarchy=data/

python src/hierarchical_eval.py --labels=data/label.json \
--mention=mention_list.txt --prediction=...
--k_parents=5 --hierarchy=data/ --merge
"""

def k_parents_eval(l_file, m_file, pred_file, k_parents, hierarchy_path, merge=False, output=None):
    """
    Args:
        l_file(str): Labels file containing dictionary of {MENTION: LABEL}.
        m_file(str): Mention list, mention per instance of the testing data.
        pred_file(str): Prediction result of format: [PRED] \tab [TRUTH] 
        k_parents(int): K-layer accuracy to be evaluated.
        hierarchy_path(str): Path to file which stores all hierarchical paths
                            containing {MENTION: PATHS}.
        merge(bool): Evaluation mode, choice: [MERGE, RAW]
        output(str): Output filename. [NOT_IN_USE_NOW]
    """

    # Load all mentions in the testing data
    eval_mention = readlines(m_file)
    test_mentions = np.unique(eval_mention)
    # Load k-parents tree
    hierarchy_dict = merge_dict(hierarchy_path, postfix="_kptree.json")

    # Get paths of all unique mentions
    test_paths = [hierarchy_dict[itr] for itr in test_mentions]
    # Unpack paths (flatten)
    test_paths = list(chain.from_iterable(test_paths))
    # Statistics about the path depth: {depth: number}
    depth_stat = Counter([len(itr) for itr in test_paths])
    # Find the depth of all paths
    max_depth = max(depth_stat.keys())

    # Calculate the nodes in the testing partial tree
    statistics = list()
    for itr in range(max_depth):
        nodes_in_layer = list()
        # Collect nodes in the same layer
        for p in test_paths:
            try:
                nodes_in_layer.append(p[itr])
            except:
                continue
        # Unique those nodes in the same layer
        nodes_in_layer = np.unique(nodes_in_layer)
        # print(nodes_in_layer)
        statistics.append(nodes_in_layer.size)
    # print(statistics)

    # Load label mapping
    labels_dict = json.load(open(l_file, "r"))
    # Reverse label mapping to text content
    labels_dict = {val: key for key, val in labels_dict.items()}
    # Load result for eval
    pred = readlines(pred_file)
    # [Prediction] \tab [Ground_Truth]
    pred = [itr.split("\t") for itr in pred]
    # Multi-label using comma as separator
    pred = [[itr[0].split(","), itr[1].split(",")] for itr in pred]

    # Parse str to integer and lookup for the type (in context)
    print("\nCalculating {:d}-layer parents accuracy (MODE: {:s}):"
          .format(k_parents, "MERGE" if merge else "RAW"))
    acc_collection, depth = list(), list()
    # For each instance
    for idx, itr in enumerate(pred):
        # Unpack prediction and ground_truth
        prediction, ground_truth = itr
        # Acquire paths from parents dictionary
        paths = hierarchy_dict[eval_mention[idx]]
        # Record all paths lengths (a.k.a the depth of a leaf node from root.)
        depth.append([len(itr) for itr in paths])

        # Lookup the dictionary containing the path of each node
        # Each entry contains list(All paths) of list(Path)
        prediction = [-1 if itr == "" else labels_dict[int(itr)] for itr in prediction]
        ground_truth = [labels_dict[int(itr)] for itr in ground_truth]

        # Calculate accuracy
        per_inst_acc = list()
        # For each path of the instance
        for itr_path in paths:
            per_path_acc = list()
            for itr_k in range(k_parents):
                # print("Layer {:d}: {:s}".format(itr_k, itr_path[itr_k]))
                try:
                    per_path_acc.append(1. if itr_path[itr_k] in prediction else 0.)
                # The result if NaN if the path is not long enough
                except:
                    per_path_acc.append(-1. if merge else np.nan)

            # Append result according to the mode
            if merge:
                per_inst_acc.append(per_path_acc)
            else:
                acc_collection.append(per_path_acc)

        # If the prediction predict correctly on some layer k, we give it 100% accuracy
        if merge:
            per_inst_acc = np.array(per_inst_acc).max(axis=0)
            per_inst_acc[per_inst_acc == -1.] = np.nan
            acc_collection.append(per_inst_acc)

    # Calculate accuracy with respect to batch and ignore NaNs
    acc_collection = np.array(acc_collection)
    accuracy = np.nanmean(acc_collection, axis=0)
    # Calculate number of instance across all classes
    n_instances = np.count_nonzero(~np.isnan(acc_collection), axis=0)
    # print(accuracy)
    print(" LAYER# | N_INSTANCES | ACCURACY | #NODES")
    print("=========================================")
    for itr, n in zip(range(k_parents), n_instances):
        print("Layer {:2d}:  {:10d} |   {:.2f}% | {:5d}"
              .format(itr + 1, n, 100. * accuracy[itr], statistics[itr]))
    # info
    depth = np.array(list(chain.from_iterable(depth)))
    print("=========================================")
    print("Depth: MEAN={:2.2f} | MAX={:2.2f} | MIN={:2.2f}"
          .format(depth.mean(), depth.max(), depth.min()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", help="Label for mapping types to label values.")
    parser.add_argument("--mention", help="Mention of each instance, for evaluation.")
    parser.add_argument("--output", help="Sentences with key words")
    parser.add_argument("--prediction", help="Dumped prediction.")
    parser.add_argument("--k_parents", type=int, default=5, help="Dumped prediction.")
    parser.add_argument("--hierarchy", help="Hierarchy information for k-parent evaluation.")
    parser.add_argument("--merge", action="store_true", help="Calculate MERGED accuracy: "
                        "If an instance's type is correctly predicted on at least one path "
                        "at layer k, than we'll count it as a correct prediction at that "
                        "layer (the true accuracy is aprroximated.)")

    args = parser.parse_args()

    k_parents_eval(args.labels, args.mention, args.prediction, args.k_parents,
                   args.hierarchy, args.merge, args.output)
