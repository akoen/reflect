import pandas as pd
import csv

import convokit
from convokit import Corpus, download, TextParser, Utterance, Speaker, Conversation
from convokit import PolitenessStrategies

# CRAFT imports
from convokit import Forecaster
from convokit.forecaster.CRAFTModel import CRAFTModel
from convokit.forecaster.CRAFT import craft_tokenize

MAX_LENGTH = 80

def load_corpus(corpus_name: str) -> "corpus":
    """Load the given corpus"""
    data_corpus = Corpus(download(corpus_name))
    data_corpus.load_info('utterance', ['parsed'])

    data_corpus.print_summary_stats()

    return data_corpus

def train_model(corpus: Corpus):
    """Train CRAFT model on data corpus"""

    craft_model = CRAFTModel(device_type='cpu', model_path='finetuned_model.tar')

    forecaster = Forecaster(forecaster_model=craft_model,
                            forecast_mode="future",
                            convo_structure="linear",
                            text_func=lambda utt: utt.meta["tokens"][:(MAX_LENGTH - 1)],
                            forecast_attribute_name="prediction", forecast_prob_attribute_name="pred_score",
                            use_last_only=False,
                            skip_broken_convos=False
                            )

    # Tokenize corpus with results from trained model
    for utt in corpus.iter_utterances():
        utt.add_meta("tokens", craft_tokenize(craft_model.voc, utt.text))

    return craft_model, forecaster


def create_corpus(sample_corpus):
    # Get random prompt from sample_corpus
    utt_prompt = sample_corpus.random_utterance()
    prompt_speaker = Speaker(id="prompt_speaker")
    utt_prompt = Utterance(id='prompt', text=utt_prompt.text, speaker=prompt_speaker, conversation_id='0',
                           reply_to=None,
                           timestamp=0)
    # Prompt for reponse
    print(utt_prompt.text)
    input_text = input("Please enter a response: ")

    # Make new corpus with prompt and response
    reply_speaker = Speaker(id="reply_speaker")
    utt_reply = Utterance(id="reply", text=input_text, speaker=reply_speaker, conversation_id='0', reply_to='prompt',
                          timestamp=1)
    convo_corpus = Corpus(utterances=[utt_prompt, utt_reply])

    # Parse text of new corpus
    ts = TextParser()
    convo_corpus = ts.transform(convo_corpus)
    print(utt_reply)

    return convo_corpus


def extract_politeness_features(corpus):
    """Extract politeness strategies features"""

    ps = PolitenessStrategies(verbose=1000)
    corpus = ps.transform(corpus)

    # Append politeness strategies to corpus
    utterance_ids = corpus.get_utterance_ids()
    rows = []
    for uid in utterance_ids:
        rows.append(corpus.get_utterance(uid).meta["politeness_strategies"])
    politeness_strategies = pd.DataFrame(rows, index=utterance_ids)
    print(politeness_strategies)

    return corpus


def calculate_derailment_probability(craft_model: CRAFTModel, forecaster: Forecaster, test_corpus: Corpus):
    """Detect whether conversation will turn awry based on training data"""

    # Tokenize test corpus
    for utt in test_corpus.iter_utterances():
        utt.add_meta("tokens", craft_tokenize(craft_model.voc, utt.text))
        print(utt)

    pred = forecaster.transform(test_corpus)

    forecasts_df = forecaster.summarize(pred)
    print(forecasts_df.loc['reply'])


def main(verbose: ('verbose output', 'flag', 'v')):
    """A tool to parse online comment responses"""
    is_verbose = verbose
    data_corpus = load_corpus('conversations-gone-awry-corpus')
    craft_model, forecaster = train_model(data_corpus)
    while(True):
        convo_corpus = create_corpus(data_corpus)
        convo_corpus = extract_politeness_features(convo_corpus)
        calculate_derailment_probability(craft_model, forecaster, convo_corpus)


if __name__ == "__main__":
    import plac
    plac.call(main)
