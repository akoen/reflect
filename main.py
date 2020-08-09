import pandas as pd
import csv
import re

import convokit
from convokit import Corpus, download, TextParser, Utterance, Speaker, Conversation
from convokit import PolitenessStrategies

# CRAFT imports
from convokit import Forecaster
from convokit.forecaster.CRAFTModel import CRAFTModel
from convokit.forecaster.CRAFT import craft_tokenize

MAX_LENGTH = 80


def load_corpus(corpus_name: str):
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
                            label_func=lambda utt: int(utt.meta['comment_has_personal_attack']),
                            forecast_attribute_name="prediction", forecast_prob_attribute_name="pred_score",
                            use_last_only=False,
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

def get_random_converstation_corpus(data_corpus: Corpus):
    convo = data_corpus.random_conversation()
    utterance_list = convo.get_chronological_utterance_list()

    new_corpus = Corpus(utterances=utterance_list[1:3])
    for utt in new_corpus.iter_utterances():
        print(utt.text, end='\n\n')

    return new_corpus


def analyze_politeness_features(corpus):
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

def get_politeness_feature_list(utt: Utterance):
    """Return a list of politeness features in human-readable format from an utterance."""

    strategies = []
    for i in utt.meta['politeness_strategies'].items():
        if i[1] == 1:
            match = re.findall(r"(?<===).*?(?===)", i[0])
            strategies.append(match[0])

    return strategies


def calculate_derailement_probability(craft_model: CRAFTModel, forecaster: Forecaster, test_corpus: Corpus):
    """Detect whether conversation will turn awry based on training data"""

    # Tokenize test corpus
    for utt in test_corpus.iter_utterances():
        utt.add_meta("tokens", craft_tokenize(craft_model.voc, utt.text))

    return forecaster.transform(test_corpus)

    #forecasts_df = forecaster.summarize(pred)
    #print(forecasts_df)


def main(verbose: ('verbose output', 'flag', 'v')):
    """A tool to parse online comment responses"""
    is_verbose = verbose
    data_corpus = load_corpus('conversations-gone-awry-corpus')
    craft_model, forecaster = train_model(data_corpus)

    with open('output-data.csv', mode='w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        for i in range(50):
            csv_row = []

            # convo_corpus = create_corpus(data_corpus)
            convo_corpus = get_random_converstation_corpus(data_corpus)
            convo_corpus = analyze_politeness_features(convo_corpus)
            convo_corpus = calculate_derailement_probability(craft_model, forecaster, convo_corpus)

            convo_ids = convo_corpus.get_utterance_ids()
            prompt = convo_corpus.get_utterance(convo_ids[0])
            reply = convo_corpus.get_utterance(convo_ids[1])
            reply_strategies = get_politeness_feature_list(reply)

            if(len(prompt.text) < 400 and len(reply.text) < 400):
                csv_row.append(prompt.text)
                csv_row.append(reply.text)
                csv_row.append(' '.join(reply_strategies))
                csv_row.append(reply.meta['pred_score'])
                csv_row.append(prompt)
                csv_row.append(reply)

                writer.writerow(csv_row)



if __name__ == "__main__":
    import plac
    plac.call(main)
