"""
BackTranscription class
-----------------------------------

"""

from transformers import WhisperForConditionalGeneration, WhisperProcessor

from textattack.shared import AttackedText

from .sentence_transformation import SentenceTransformation


class BackTranscription(SentenceTransformation):
    """A type of sentence level transformation that takes in a text input, converts it into
    synthesized speech using ASR, and transcribes it back to text using TTS.

    tts_model: text-to-speech model from huggingface
    asr_model: automatic speech recognition model from huggingface

    (!) Python libraries `fairseq`, `g2p_en` and `librosa` should be installed.

    Example::

        >>> from textattack.transformations.sentence_transformations import BackTranscription
        >>> from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
        >>> from textattack.augmentation import Augmenter

        >>> transformation = BackTranscription()
        >>> constraints = [RepeatModification(), StopwordModification()]
        >>> augmenter = Augmenter(transformation = transformation, constraints = constraints)
        >>> s = 'What on earth are you doing here.'

        >>> augmenter.augment(s)

    You can find more about the back transcription method in the following paper:

        @inproceedings{kubis-etal-2023-back,
            title = "Back Transcription as a Method for Evaluating Robustness of Natural Language Understanding Models to Speech Recognition Errors",
            author = "Kubis, Marek  and
                Sk{\\'o}rzewski, Pawe{\\l}  and
                Sowa{\\'n}nski, Marcin  and
                Zietkiewicz, Tomasz",
            editor = "Bouamor, Houda  and
                Pino, Juan  and
                Bali, Kalika",
            booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
            month = dec,
            year = "2023",
            address = "Singapore",
            publisher = "Association for Computational Linguistics",
            url = "https://aclanthology.org/2023.emnlp-main.724",
            doi = "10.18653/v1/2023.emnlp-main.724",
            pages = "11824--11835",
        }
    """

    def __init__(
        self,
        tts_model="facebook/fastspeech2-en-ljspeech",
        asr_model="openai/whisper-base",
    ):
        # TTS model
        from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
        from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

        self.tts_model_name = tts_model
        models, cfg, self.tts_task = load_model_ensemble_and_task_from_hf_hub(
            self.tts_model_name,
            arg_overrides={"vocoder": "hifigan", "fp16": False},
        )
        self.tts_model = models[0]
        TTSHubInterface.update_cfg_with_data_cfg(cfg, self.tts_task.data_cfg)
        self.tts_generator = self.tts_task.build_generator(models, cfg)

        # ASR model
        self.asr_model_name = asr_model
        self.asr_sampling_rate = 16000
        self.asr_processor = WhisperProcessor.from_pretrained(self.asr_model_name)
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(
            self.asr_model_name
        )
        self.asr_model.config.forced_decoder_ids = None

    def back_transcribe(self, text):
        # speech synthesis
        from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

        sample = TTSHubInterface.get_model_input(self.tts_task, text)
        wav, rate = TTSHubInterface.get_prediction(
            self.tts_task, self.tts_model, self.tts_generator, sample
        )

        # speech recognition
        import librosa

        resampled_wav = librosa.resample(
            wav.numpy(), orig_sr=rate, target_sr=self.asr_sampling_rate
        )
        input_features = self.asr_processor(
            resampled_wav, sampling_rate=self.asr_sampling_rate, return_tensors="pt"
        ).input_features

        predicted_ids = self.asr_model.generate(input_features)

        transcription = self.asr_processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        return transcription[0].strip()

    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []
        current_text = current_text.text

        # do the back transcription
        back_transcribed_text = self.back_transcribe([current_text])

        transformed_texts.append(AttackedText(back_transcribed_text))
        return transformed_texts
