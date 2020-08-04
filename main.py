from convokit import Corpus, download
# corpus = Corpus(filename=download("conversations-gone-awry-cmv-corpus"))
cornell_corpus = Corpus(filename=download("subreddit-Cornell"))

print(cornell_corpus)
cornell_corpus.print_summary_stats()
