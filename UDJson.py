# Формирование набора данных в формате JSON из выгрузки Толоки
# Для морфологического и синтаксического анализа используется UDpipe

import pandas as pd
import json
import argparse as ap
import ufal.udpipe
import conllu
import tqdm

class Model:
    def __init__(self, path):
        """Load given model."""
        self.model = ufal.udpipe.Model.load(path)
        if not self.model:
            raise Exception("Cannot load UDPipe model from file '%s'" % path)

    def tokenize(self, text):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text, tokenizer)

    def read(self, text, in_format):
        """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
        input_format = ufal.udpipe.InputFormat.newInputFormat(in_format)
        if not input_format:
            raise Exception("Cannot create input format '%s'" % in_format)
        return self._read(text, input_format)

    def _read(self, text, input_format):
        input_format.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.model.tag(sentence, self.model.DEFAULT)

    def parse(self, sentence):
        """Parse the given ufal.udpipe.Sentence (inplace)."""
        self.model.parse(sentence, self.model.DEFAULT)

    def write(self, sentences, out_format):
        """Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""

        output_format = ufal.udpipe.OutputFormat.newOutputFormat(out_format)
        output = ''
        for sentence in sentences:
            output += output_format.writeSentence(sentence)
        output += output_format.finishDocument()

        return output


def get_feats_string(feats, sep="|"):
    if feats is not None:
        l_res = []
        for k, v in feats.items():
            l_res.append("{0}={1}".format(k, v))
        res = sep.join(l_res)
    else:
        res = "_"
    return res


def text_to_json(text, model, sep="|"):
    """

    Parameters
    ----------
    text: str
    model: Model
    sep: str

    Returns
    -------
    l_sentences: list
    """

    segmented = ufal.udpipe.Pipeline(model.model, "tokenize", ufal.udpipe.Pipeline.NONE, ufal.udpipe.Pipeline.NONE, "").process(text)
    sentences = model.read(segmented, "conllu")
    for sent in sentences:
        model.tag(sent)
        model.parse(sent)
    res_conllu = model.write(sentences, "conllu")

    tmp_text = str(text)
    posStart_prev = 0
    l_sentences = []
    for sent in conllu.parse(res_conllu):
        l_sent = []
        for word in sent:
            posStart = str.find(tmp_text, word["form"], posStart_prev)
            posStart_prev = posStart
            d_word = {"id": word["id"],
                      "forma": word["form"],
                      "lemma": word["lemma"],
                      "pos": word["upostag"],
                      "grm": get_feats_string(word["feats"], sep=sep),
                      "len": len(word["form"]),
                      "posStart": posStart,
                      "dom": word["head"],
                      "link": word["deprel"]}
            l_sent.append(d_word)
        l_sentences.append(l_sent)

    return l_sentences


if __name__ == "__main__":
    args_parser = ap.ArgumentParser()
    args_parser.add_argument("inp_csv", help="Input")
    args_parser.add_argument("result")
    args_parser.add_argument("udpipe_model")
    args_parser.add_argument("--grm-sep", dest="grm_sep", default="|")
    args_parser.add_argument("--csv-sep", default=";")
    args = args_parser.parse_args()

    udpipe_model = Model(args.udpipe_model)
    inp = pd.read_csv(args.inp_csv, sep=args.csv_sep)
    inp = inp.dropna(axis=1)
    with tqdm.tqdm(total=len(inp)) as pbar:
        l_res = []
        for row_ind, row in inp.iterrows():
            d_doc = {"meta": {}}
            for column_name, column_value in row.items():
                if column_name != "text":
                    d_doc["meta"][column_name] = column_value
                else:
                    d_doc["text"] = column_value
                    d_doc["sentences"] = text_to_json(column_value, udpipe_model, sep=args.grm_sep)
            l_res.append(d_doc)
            pbar.update(1)

    with open(args.result, "w") as f:
        json.dump(l_res, f, indent=4, ensure_ascii=False, sort_keys=True, default=float)

    print("Done!")