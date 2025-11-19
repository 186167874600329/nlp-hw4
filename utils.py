import nltk
nltk.data.path.append('/scratch/yy5074/nltk_data/tokenizers/punkt/PY3')
nltk.data.path.append('/scratch/yy5074/nltk_data')



import random
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer



# 这个是没用的模板函数（不删也没关系）
def example_transform(example):
    example["text"] = example["text"].lower()
    return example


def custom_transform(example):
    text = example["text"]
    words = word_tokenize(text)
    detok = TreebankWordDetokenizer()

    def synonym_or_typo(word):
        # 跳过标点或数字
        if not word.isalpha():
            return word

        # 🔹10% 几率进行同义词替换
        if random.random() < 0.25:
            synsets = wordnet.synsets(word)
            if synsets:
                lemmas = [l.name().replace("_", " ") for l in synsets[0].lemmas()]
                # 只取不同的词
                for lemma in lemmas:
                    if lemma.lower() != word.lower():
                        return lemma

        # 🔹10% 几率制造拼写错误（交换相邻字母）
        if random.random() < 0.25 and len(word) > 3:
            i = random.randint(0, len(word) - 2)
            w_list = list(word)
            w_list[i], w_list[i + 1] = w_list[i + 1], w_list[i]
            return "".join(w_list)

        # 否则返回原单词
        return word

    # 对句子中每个词做替换/扰动
    new_words = [synonym_or_typo(w) for w in words]
    example["text"] = detok.detokenize(new_words)
    return example
