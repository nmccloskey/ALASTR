import re
import os
import numpy as np
from dendropy import Tree
from collections import Counter
import logging
logger = logging.getLogger("CustomLogger")
from nltk import Tree as nltkTree
from alastr.backend.nlp.NLPmodel import NLPmodel
from zss import simple_distance, Node
from dendropy import Tree, TaxonNamespace
from dendropy.calculate import treecompare
from alastr.analyses.morphology import analyze_spacy_features
from alastr.backend.eda.visualization import make_spacy_dep_pdfs


def compute_tree_height(token):
    """Recursively compute the height of the syntactic tree."""
    if not list(token.children):  # If leaf node
        return 1
    return 1 + max(compute_tree_height(child) for child in token.children)

def spacy_to_zss_tree(token):
    """ Convert a SpaCy dependency tree to a ZSS-compatible tree. """
    node = Node(token.text)
    for child in token.children:
        node.addkid(spacy_to_zss_tree(child))
    return node

def constituency_to_zss(tree):
    """Convert an NLTK constituency tree to ZSS format."""
    if isinstance(tree, str):
        return Node(tree)
    node = Node(tree.label())
    for child in tree:
        node.addkid(constituency_to_zss(child))
    return node

def sent_to_newick(sent):
    """Convert a SpaCy Benepar constituency parse into a valid Newick string with sequential numeric node labels."""
    NLP = NLPmodel()
    nlp = NLP.get_nlp()
    new_text = re.sub(r'[^\w\s]','',sent.text)
    new_doc = nlp(new_text)
    sent = next(new_doc.sents)

    sent_text = sent._.parse_string  # Get the constituency tree string
    
    # Remove unwanted punctuation nodes
    sent_text = re.sub(r'\s\([\.\!\?,;]\s[\.\!\?,;]\)', '', sent_text)

    # Convert Penn Treebank format to an NLTK Tree object
    tree = nltkTree.fromstring(sent_text)

    # Step 1: Convert tree to Newick format with words
    def tree_to_newick(node):
        """Recursively converts an NLTK Tree into a Newick string."""
        if isinstance(node, str):  # If it's a terminal node, return it
            return node
        else:
            children = ",".join(tree_to_newick(child) for child in node)
            return f"({children})"

    newick = "[&R] " + tree_to_newick(tree) + ";"

    # Step 2: Replace each token with a unique index using re.finditer()
    def replace_with_indices(newick_str):
        """Replace tokens sequentially with incremental numbers."""
        token_matches = list(re.finditer(r'\w+', newick_str))  # Find all tokens
        newick_copy = newick_str  # Work on a copy

        for i, match in enumerate(token_matches):
            token = match.group()  # Extract the actual token
            newick_copy = newick_copy.replace(token, str(i), 1)  # Replace only the first occurrence
        
        return newick_copy

    return replace_with_indices(newick)

def run_treecompare(sent1, sent2):
    """
    Compute symmetric difference between two constituency trees in Newick format.
    
    Args:
        sent1 (spacy.tokens.Span): First sentence.
        sent2 (spacy.tokens.Span): Second sentence.

    Returns:
        int: The symmetric difference between two trees.
    """
    n1 = sent_to_newick(sent1)
    n2 = sent_to_newick(sent2)

    taxa = TaxonNamespace()

    t1 = Tree.get(data=n1, schema="newick", taxon_namespace=taxa, suppress_internal_node_taxa=True, suppress_leaf_node_taxa=False)
    t2 = Tree.get(data=n2, schema="newick", taxon_namespace=taxa, suppress_internal_node_taxa=True, suppress_leaf_node_taxa=False)
    
    b1 = t1.encode_bipartitions()
    b2 = t2.encode_bipartitions()

    return treecompare.symmetric_difference(t1, t2)

def compare_trees(doc):
    """
    Compute syntactic similarity matrices using Tree Edit Distance (TED) and Symmetric Distance (SD).
    
    Args:
        doc (spacy.tokens.Doc): A processed SpaCy document with sentence boundaries.

    Returns:
        dict: Computed syntactic diversity metrics.
    """
    sents = list(doc.sents)
    n = len(sents)
    func_data = {}
    func_data["tree_comp"] = {"total_sents": n}

    # Initialize distance matrices
    dep_zssdmat = np.zeros((n, n))  # Dependency Tree Edit Distance
    tree_zssdmat = np.zeros((n, n))  # Constituency Tree Edit Distance
    tree_sdmat = np.zeros((n, n))  # Constituency Symmetric Distance

    for i in range(n):
        for j in range(i + 1, n):

            # Compute Dependency Tree Edit Distance (TED)
            zss1 = spacy_to_zss_tree(sents[i].root)
            zss2 = spacy_to_zss_tree(sents[j].root)
            dist_zss_dep = simple_distance(zss1, zss2)
            dep_zssdmat[i, j] = dist_zss_dep
            dep_zssdmat[j, i] = dist_zss_dep

            # Compute Constituency Tree Edit Distance (TED)
            zss1 = spacy_to_zss_tree(sents[i].root)
            zss2 = spacy_to_zss_tree(sents[j].root)
            dist_zss_tree = simple_distance(zss1, zss2)
            tree_zssdmat[i, j] = dist_zss_tree
            tree_zssdmat[j, i] = dist_zss_tree

            # Compute Constituency Tree Symmetric Difference (SD)
            dist_sd = run_treecompare(sents[i], sents[j])
            tree_sdmat[i, j] = dist_sd
            tree_sdmat[j, i] = dist_sd

    for lab, mat in zip(["dep_zss", "tree_zss", "tree_sdmat"], [dep_zssdmat, tree_zssdmat, tree_sdmat]):
        nonzero_values = mat[np.nonzero(mat)]  # Extract only non-zero values
        sent_lengths = np.array([len(sent) for sent in sents])  # Sentence lengths
        
        if len(nonzero_values) == 0:  # Avoid division by zero
            func_data["tree_comp"].update({f"{lab}_min": None, f"{lab}_max": None, f"{lab}_mean": None, f"{lab}_median": None,
                            f"{lab}_var": None, f"{lab}_std_dev": None, f"{lab}_cv": None,
                            f"{lab}_weighted_mean_dist": None, f"{lab}_normalized_diversity": None})
        else:
            # Ensure weights match nonzero_values
            valid_weights = sent_lengths[:len(nonzero_values)]
            if len(valid_weights) != len(nonzero_values):
                valid_weights = np.ones_like(nonzero_values)  # Default to uniform weights if shape mismatch

            func_data["tree_comp"].update({
                f"{lab}_min": np.min(nonzero_values),
                f"{lab}_max": np.max(nonzero_values),
                f"{lab}_mean": np.mean(nonzero_values),
                f"{lab}_median": np.median(nonzero_values),
                f"{lab}_var": np.var(nonzero_values),
                f"{lab}_std_dev": np.std(nonzero_values),
                f"{lab}_cv": np.std(nonzero_values) / np.mean(nonzero_values),
                f"{lab}_weighted_mean_dist": np.average(nonzero_values, weights=valid_weights),
                f"{lab}_normalized_diversity": np.mean(nonzero_values) / (np.mean(valid_weights) if np.mean(valid_weights) != 0 else 1)
            })

    return func_data

def analyze_syntactic_trees(doc):
    """
    Analyze syntactic tree structure from a text using SpaCy.

    Args:
        doc (spacy.Doc): Processed text document.

    Returns:
        dict: Extracted syntactic features.
    """

    logger.info("Starting syntactic tree analysis...")

    # Initialize result storage
    func_data = {}
    func_data["syn_trees"] = {
        "max_tree_height": 0,
        "min_tree_height": float('inf'),
        "avg_tree_height": 0,
        "num_root_tokens": 0,
        # "root_words": Counter(),
        "num_clauses": 0,
        "num_NP": 0,
        "num_VP": 0,
        "num_PP": 0,
        "num_unique_deps": 0,
        "most_common_dep": None,
        "subj_to_root_distance_avg": 0,
        "depth_per_token_avg": 0
    }

    tree_heights = []
    subject_distances = []
    depth_per_token = []
    dep_counts = Counter()
    branch_counts = Counter()
    total_children = 0
    total_nodes = 0
    conjunction_data = Counter()
    negation_data = Counter()
    total_negations = 0
    not_count = 0
    contracted_neg_count = 0

    for sent in doc.sents:
        root = sent.root
        func_data["syn_trees"]["num_root_tokens"] += 1

        # Compute tree height for this sentence
        tree_height = compute_tree_height(root)
        tree_heights.append(tree_height)

        # Compute depth per token
        for token in sent:
            depth = len(list(token.ancestors))
            depth_per_token.append(depth)

            # Track dependency relations
            dep_counts[token.dep_] += 1

            # Measure subject-to-root distance
            if "subj" in token.dep_:  
                subject_distances.append(depth)

            # Analyze branching
            num_children = len(list(token.children))
            if num_children > 0:
                branch_counts[f"n{num_children}_branchings"] += 1
                total_children += num_children
                total_nodes += 1

            # Track Coordinating Conjunctions (CC)
            if token.dep_ == "cc":  
                conjunction = token.text.upper()
                conjunction_data[f"{conjunction}_count"] += 1

                # Find the coordinated elements
                left_phrase = next(token.head.lefts, None)
                right_phrase = next(token.head.rights, None)

                if left_phrase and right_phrase:
                    left_type = left_phrase.pos_
                    right_type = right_phrase.pos_
                    phrase_pair = f"{left_type}_{right_type}"
                    conjunction_data[f"{conjunction}_{phrase_pair}"] += 1

            # Track Negations ("not", "n't")
            if token.dep_ == "neg":
                total_negations += 1
                if "'" in token.text:
                    contracted_neg_count += 1
                else:
                    not_count += 1
                negation_data[f"neg_{token.head.pos_}_count"] += 1

        # Count phrase types
        func_data["syn_trees"]["num_NP"] += sum(1 for chunk in doc.noun_chunks if chunk.root in sent)
        func_data["syn_trees"]["num_VP"] += sum(1 for token in sent if token.pos_ == "VERB" and token.dep_ == "ROOT")
        func_data["syn_trees"]["num_PP"] += sum(1 for token in sent if token.pos_ == "ADP")  
        func_data["syn_trees"]["num_clauses"] += sum(1 for token in sent if token.dep_ in {"ccomp", "advcl", "acl", "relcl"})

    # Final calculations
    if tree_heights:
        func_data["syn_trees"]["max_tree_height"] = max(tree_heights)
        func_data["syn_trees"]["min_tree_height"] = min(tree_heights)
        func_data["syn_trees"]["avg_tree_height"] = np.mean(tree_heights)

    func_data["syn_trees"]["mean_dependency_distance"] = np.mean([abs(token.i - token.head.i) for token in doc if token.head != token])
    
    if subject_distances:
        func_data["syn_trees"]["subj_to_root_distance_avg"] = np.mean(subject_distances)

    if depth_per_token:
        func_data["syn_trees"]["depth_per_token_avg"] = np.mean(depth_per_token)

    func_data["syn_trees"]["avg_branching_factor"] = total_children / total_nodes if total_nodes > 0 else 0

    # Merge all dictionaries into results
    func_data["syn_trees"].update(branch_counts)
    func_data["syn_trees"].update(conjunction_data)
    func_data["syn_trees"].update(negation_data)

    logger.info(f"Syntactic tree analysis completed.")
    return func_data

def analyze_syntax(PM, sample_data):

    try:
        results = PM.sections["syntax"].init_results_dict()

        NLP = NLPmodel()
        nlp = NLP.get_nlp()

        if PM.sentence_level:
            if not isinstance(sample_data, list):
                raise ValueError("Expected a list of sentence dicts for sentence-level analysis.")
            
            doc_cleaned = ""
            doc_id = sample_data[0].get("doc_id")            
            
            for sent in sample_data:
                sent_id = sent.get("sent_id")
                cleaned = sent.get("cleaned", "")

                doc = nlp(cleaned)
                sent_data_base = {"doc_id": doc_id, "sent_id": sent_id}
                func_data = analyze_syntactic_trees(doc)
                func_data.update(analyze_spacy_features(doc, 5, "DEP"))

                for table, row_data in func_data.items():
                    sent_data = sent_data_base.copy()
                    sent_data.update(row_data)
                    results[f"{table}_sent"].append(sent_data)

                doc_cleaned += " " + cleaned

            doc_cleaned = doc_cleaned.strip()

        else:
            if not isinstance(sample_data, dict):
                raise ValueError("Expected a single dict for document-level analysis.")
            
            doc_cleaned = sample_data.get("cleaned", "")
            doc_id = sample_data.get("doc_id")

        doc_data_base = {"doc_id": doc_id}
        doc = nlp(doc_cleaned)
        func_data = analyze_syntactic_trees(doc)
        func_data.update(analyze_spacy_features(doc, 10, "DEP"))
        func_data.update(compare_trees(doc))

        for table, row_data in func_data.items():
            doc_data = doc_data_base.copy()
            doc_data.update(row_data)
            results[f"{table}_doc"].update(doc_data)

        if PM.dep_trees:
            path = os.path.join(PM.om.output_dir, "syntax", "doc", "dep_trees")
            logger.info(f"Saving dep trees to {path}")
            make_spacy_dep_pdfs(doc, doc_id, path)

        logger.info(f"Syntactic analysis completed successfully.")
        return results

    except Exception as e:
        logger.error(f"Error analyzing syntax: {e}")
        return {}
