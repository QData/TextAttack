"""
BackTranslation class
-----------------------------------

"""


import random

from transformers import MarianMTModel, MarianTokenizer

from textattack.shared import AttackedText

from .sentence_transformation import SentenceTransformation


class BackTranslation(SentenceTransformation):
    """A type of sentence level transformation that takes in a text input,
    translates it into target language and translates it back to source
    language.

    letters_to_insert (string): letters allowed for insertion into words
    (used by some char-based transformations)

    src_lang (string): source language
    target_lang (string): target language, for the list of supported language check bottom of this page
    src_model: translation model from huggingface that translates from source language to target language
    target_model: translation model from huggingface that translates from target language to source language
    chained_back_translation: run back translation in a chain for more perturbation (for example, en-es-en-fr-en)

    Example::

        >>> from textattack.transformations.sentence_transformations import BackTranslation
        >>> from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
        >>> from textattack.augmentation import Augmenter

        >>> transformation = BackTranslation()
        >>> constraints = [RepeatModification(), StopwordModification()]
        >>> augmenter = Augmenter(transformation = transformation, constraints = constraints)
        >>> s = 'What on earth are you doing here.'

        >>> augmenter.augment(s)
    """

    def __init__(
        self,
        src_lang="en",
        target_lang="es",
        src_model="Helsinki-NLP/opus-mt-ROMANCE-en",
        target_model="Helsinki-NLP/opus-mt-en-ROMANCE",
        chained_back_translation=0,
    ):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.target_model = MarianMTModel.from_pretrained(target_model)
        self.target_tokenizer = MarianTokenizer.from_pretrained(target_model)
        self.src_model = MarianMTModel.from_pretrained(src_model)
        self.src_tokenizer = MarianTokenizer.from_pretrained(src_model)
        self.chained_back_translation = chained_back_translation

    def translate(self, input, model, tokenizer, lang="es"):
        # change the text to model's format
        src_texts = []
        if lang == "en":
            src_texts.append(input[0])
        else:
            if ">>" and "<<" not in lang:
                lang = ">>" + lang + "<< "
            src_texts.append(lang + input[0])

        # tokenize the input
        encoded_input = tokenizer.prepare_seq2seq_batch(src_texts, return_tensors="pt")

        # translate the input
        translated = model.generate(**encoded_input)
        translated_input = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return translated_input

    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []
        current_text = current_text.text

        # to perform chained back translation, a random list of target languages are selected from the provided model
        if self.chained_back_translation:
            list_of_target_lang = random.sample(
                self.target_tokenizer.supported_language_codes,
                self.chained_back_translation,
            )
            for target_lang in list_of_target_lang:
                target_language_text = self.translate(
                    [current_text],
                    self.target_model,
                    self.target_tokenizer,
                    target_lang,
                )
                src_language_text = self.translate(
                    target_language_text,
                    self.src_model,
                    self.src_tokenizer,
                    self.src_lang,
                )
                current_text = src_language_text[0]
            return [AttackedText(current_text)]

        # translates source to target language and back to source language (single back translation)
        target_language_text = self.translate(
            [current_text], self.target_model, self.target_tokenizer, self.target_lang
        )
        src_language_text = self.translate(
            target_language_text, self.src_model, self.src_tokenizer, self.src_lang
        )
        transformed_texts.append(AttackedText(src_language_text[0]))
        return transformed_texts


"""
List of supported languages
['fr',
 'es',
 'it',
 'pt',
 'pt_br',
 'ro',
 'ca',
 'gl',
 'pt_BR<<',
 'la<<',
 'wa<<',
 'fur<<',
 'oc<<',
 'fr_CA<<',
 'sc<<',
 'es_ES',
 'es_MX',
 'es_AR',
 'es_PR',
 'es_UY',
 'es_CL',
 'es_CO',
 'es_CR',
 'es_GT',
 'es_HN',
 'es_NI',
 'es_PA',
 'es_PE',
 'es_VE',
 'es_DO',
 'es_EC',
 'es_SV',
 'an',
 'pt_PT',
 'frp',
 'lad',
 'vec',
 'fr_FR',
 'co',
 'it_IT',
 'lld',
 'lij',
 'lmo',
 'nap',
 'rm',
 'scn',
 'mwl']
"""
