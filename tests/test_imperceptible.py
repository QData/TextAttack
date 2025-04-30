import textattack
from textattack.models.wrappers import ModelWrapper
from typing import List
from transformers import pipeline
from datasets import load_dataset
from string import punctuation
import argparse
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import os
import tarfile
import requests
from io import BytesIO
from bs4 import BeautifulSoup
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", choices=["emotion", "ner", "translation", "toxic", "mnli"], required=True, help="Choose which experiment to run.")
parser.add_argument("--perturbation_type", choices=["homoglyphs", "invisible", "deletions", "reorderings"], required=True, help="Choose which perturbation type to use.")
args = parser.parse_args()

"""
Emotion targeted attack
"""

if args.experiment == "emotion":

    class EmotionWrapper(ModelWrapper):

        def __init__(self, model):
            self.model = model

        def __call__(self, input_texts: List[str]) -> List[List[float]]:

            """
            Args:
                input_texts: List[str]

            Return:
                ret: List[List[float]]
                a list of elements, one per element of input_texts. Each element is a list of probabilities, one for each label.
            """
            ret = []
            for i in input_texts:
                pred = self.model(i)[0]
                scores = []
                for j in pred:
                    scores.append(j['score'])
                ret.append(scores)
            return ret

    model = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True, device=-1)
    model_wrapper = EmotionWrapper(model)

    attack = textattack.attack_recipes.BadCharacters2021.build(
        model_wrapper, 
        goal_function_type="targeted_strict",
        perturbation_type=args.perturbation_type
    )
    dataset = textattack.datasets.HuggingFaceDataset("emotion", split="test")
    print(dataset[0])
    attacker = textattack.Attacker(attack, dataset)
    attacker.attack_dataset()

elif args.experiment == "ner":
    class NERModelWrapper(ModelWrapper):
        def __init__(self, model):
            self.model = model

        def __call__(self, input_texts: List[str]):
            """
            Args:
                input_texts: List[str]
            
            Return:
                ret
                    Model output
            """
            ret = []
            for i in input_texts:
                pred = self.model(i)
                ret.append(pred)
            return ret

    model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    model_wrapper = NERModelWrapper(model)
    ner_classes = ['PER', 'ORG', 'LOC', 'MISC']
    attack = textattack.attack_recipes.BadCharacters2021.build(
        model_wrapper, 
        goal_function_type="named_entity_recognition", 
        perturbation_type=args.perturbation_type, 
        target_suffix=ner_classes[0]
    )
    dataset = load_dataset("conll2003", split="test", trust_remote_code=True)
    pairs = []
    def detokenize(tokens: List[str]) -> str:
        output = ""
        for index, token in enumerate(tokens):
            if (len(token) == 1 and token in punctuation) or index == 0:
                output += token
            else:
                output += ' ' + token
        return output
    for ex in dataset:
        tokens = ex["tokens"]
        ner_labels = ex["ner_tags"]
        text = detokenize(tokens) 
        pairs.append((text, "NER")) # hack
    dataset = textattack.datasets.Dataset(pairs)
    attacker = textattack.Attacker(attack, dataset)
    attacker.attack_dataset()

elif args.experiment == "translation":
    class FairseqTranslationWrapper(ModelWrapper):
        """
        A wrapper for the model
            torch.hub.load('pytorch/fairseq',
                            'transformer.wmt14.en-fr',
                            tokenizer='moses',
                            bpe='subword_nmt',
                            verbose=False).eval()
        or any other model with a .translate() method.
        """

        def __init__(self, model):
            self.model = model  

        def __call__(self, text_input_list: List[str]) -> List[str]:
            """
            Args:
                input_texts: List[str]
            
            Return:
                ret: List[str]
                    Result of translation. One per element in input_texts.
            """
            return [self.model.translate(text) for text in text_input_list]
    model = torch.hub.load(
        'pytorch/fairseq',
        'transformer.wmt14.en-fr',
        tokenizer='moses',
        bpe='subword_nmt',
        verbose=False
    ).eval()

    model_wrapper = FairseqTranslationWrapper(model)
    def download_en_fr_dataset():
        

        # Define constants
        url = "http://statmt.org/wmt14/test-full.tgz"
        target_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(target_dir, exist_ok=True)

        print(f"Downloading WMT14 test data from {url}...")

        # Download and extract in-memory
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with tarfile.open(fileobj=BytesIO(response.content), mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith("newstest2014-fren-src.en.sgm") or member.name.endswith("newstest2014-fren-ref.fr.sgm"):
                    print(f"Extracting {member.name} to {target_dir}")
                    member.name = os.path.basename(member.name) 
                    tar.extract(member, path=target_dir)

        print("en_fr dataset downloaded.")
    def load_en_fr_dataset():
        """
        Loads English-French sentence pairs from SGM files and returns a TextAttack Dataset.

        Returns:
            textattack.datasets.Dataset: wrapped dataset of (English, French) pairs.
        """
        

        source_path = os.path.join(os.path.dirname(__file__), "data/newstest2014-fren-src.en.sgm")
        target_path = os.path.join(os.path.dirname(__file__), "data/newstest2014-fren-ref.fr.sgm")

        with open(source_path, "r", encoding="utf-8") as f:
            source_doc = BeautifulSoup(f, "html.parser")

        with open(target_path, "r", encoding="utf-8") as f:
            target_doc = BeautifulSoup(f, "html.parser")

        pairs = []

        for doc in source_doc.find_all("doc"):
            docid = str(doc["docid"])
            for seg in doc.find_all("seg"):
                segid = str(seg["id"])
                src = str(seg.string).strip() if seg.string else ""
                tgt_node = target_doc.select_one(f'doc[docid="{docid}"] > seg[id="{segid}"]')
                if tgt_node and tgt_node.string:
                    tgt = str(tgt_node.string).strip()
                    pairs.append((src, tgt))
        return textattack.datasets.Dataset(pairs) 
    download_en_fr_dataset()
    dataset = load_en_fr_dataset()
    attack = textattack.attack_recipes.BadCharacters2021.build(
        model_wrapper, 
        goal_function_type="maximize_levenshtein", 
        perturbation_type=args.perturbation_type,
        target_distance=0.1
    )
    print(dataset[0])
    attacker = textattack.Attacker(attack, dataset)
    attacker.attack_dataset()


elif args.experiment == "toxic":
    class IBMMAXToxicWrapper(ModelWrapper):
        """
        A wrapper for the IBM Max Toxic model
        https://github.com/IBM/MAX-Toxic-Comment-Classifier/blob/master/core/model.py
        """
        def __init__(self, ibm_model_wrapper):
            """
            Args:
                ibm_model_wrapper: An instance of the IBM MAX Toxic `ModelWrapper()` class.
            """
            self.model = ibm_model_wrapper

        def __call__(self, input_text_list: List[str]) -> np.ndarray:
            """
            Args:
                input_texts: List[str]
            
            Return:
                ret: np.ndarray
                    One entry per element in input_text_list. Each is a list of logits, one for each label.
            """
            self.model._pre_process(input_text_list)
            logits = self.model._predict(input_text_list)
            return np.array(logits)
    
    

elif args.experiment == "mnli":
    class FairseqMnliWrapper(ModelWrapper):
        """
        A wrapper for the model
            torch.hub.load('pytorch/fairseq', 'roberta.large.mnli').eval()
        """

        def __init__(self, model):
            self.model = model

        def __call__(self, input_texts: List[Tuple[str, str]]) -> List[torch.Tensor]:
            """
            Args:
                input_texts: List[Tuple[str, str]]
                    List of (premise, hypothesis)
            
            Return:
                ret: List[torch.Tensor]
                    Each tensor is a list of probabilities, 
                    one for each of (contradiction, neutral, entailment)
            """
            ret = []
            for t in input_texts:
                premise = t[0]
                hypothesis = t[1]
                tokens = self.model.encode(premise, hypothesis)
                predict = self.model.predict('mnli', tokens)
                probs = softmax(predict, dim=1).cpu().detach()[0]
                ret.append(probs.unsqueeze(0))
            return ret
