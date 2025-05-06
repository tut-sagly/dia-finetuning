from datasets import load_dataset, get_dataset_config_names, interleave_datasets
from .dataset import HFDiaIterDataset


LANG_NAME_TO_CODE = {
    "dutch":      "nl",
    "french":     "fr",
    "german":     "de",
    "italian":    "it",
    "polish":     "pl",
    "portuguese": "pt",
    "spanish":    "es",
    # add more if other configs appear...
}


def load_cml_tts_streamed(dia_cfg, dac_model):
    """
    Stream all language subsets of the CML-TTS dataset in train split,
    add a `language` field, drop all except `text`, `audio`, `language`,
    and interleave them into one streaming Dataset.

    Returns:
        datasets.IterableDataset: interleaved streaming dataset
    """
    # 1) Discover all language subsets
    lang_configs = get_dataset_config_names("ylacombe/cml-tts")

    # 2) Build one streaming subset per language, with only desired columns
    streams = []
    num_ex=0
    for lang in lang_configs:
        
        iso_code = LANG_NAME_TO_CODE.get(lang, lang)
        ds_stream = load_dataset(
            "ylacombe/cml-tts",
            name=lang,
            split="train",
            streaming=True
        )

        num_ex += ds_stream.info.splits['train'].num_examples
        # keep only text, audio, and add language
        def _add_lang(ex, iso=iso_code):
            return {
                "text": ex["text"],
                "audio": ex["audio"],
                "language": iso
            }
        ds_stream = ds_stream.map(
            _add_lang,
            remove_columns=[c for c in ds_stream.column_names if c not in ["text", "audio", "language"]]
        )
        streams.append(ds_stream)

    # 3) Interleave all streams into one unified stream
    interleaved = interleave_datasets(streams, stopping_strategy="all_exhausted")
    ds = HFDiaIterDataset(interleaved, dia_cfg, dac_model)
    ds.total_examples = num_ex
    return ds