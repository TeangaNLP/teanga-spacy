import spacy
from teanga import Service

class SpacyService(Service):
    """A service that uses spaCy to tokenize and tag text.

    This service requires a SpaCY model name as a parameter. The model
    name is the name of a SpaCY model that has been installed on the
    system. For example, the model "en_core_web_sm" is a small English
    model that can be installed with the command:

    python -m spacy download en_core_web_sm

    The SpaCY model is loaded in the setup() method. The model is then
    applied to the text in the execute() method. The SpaCY model
    produces a number of annotations, including tokens, part-of-speech
    tags, lemmas, and dependency relations. These annotations are
    converted to the Teanga format and added to the document.

    Example:
    --------

    >>> from teanga import Document, Corpus
    >>> corpus = Corpus()
    >>> corpus.add_layer_meta("text")
    >>> service = SpacyService("en_core_web_sm")
    >>> service.setup()
    >>> doc = corpus.add_doc("This is a test.")
    >>> corpus.apply(service)
    """
    def __init__(self, model_name):
        """Create a service for the SpaCY model name"""
        super().__init__()
        self.model_name = model_name

    def setup(self):
        """Load the SpaCY model"""
        if not hasattr(self, "nlp") or not self.nlp:
            self.nlp = spacy.load(self.model_name)

    def requires(self):
        """Return the requirements for this service"""
        return {"text": { "type": "characters" }}

    def produces(self):
        """Return the output of this service"""
        return {
                "tokens": {"type": "span", "on": "text" },
                "pos": {"type": "seq", "on": "tokens", "data": 
                        ["ADJ","ADP","PUNCT","ADV","AUX","SYM","INTJ",
                         "CCONJ","X","NOUN","DET","PROPN","NUM","VERB",
                         "PART","PRON","SCONJ"]},
                "tag": {"type": "seq", "on": "tokens", "data": "string" },
                "lemma": {"type": "seq", "on": "tokens", "data": "string" },
                "dep": {"type": "seq", "on": "tokens",
                        "data": "links", "values": [ "acl", "acl:relcl",
                                                    "advcl", "advcl:relcl", 
                                                    "advmod", "advmod:emph", 
                                                    "advmod:lmod", "amod",
                                                    "appos", "aux",
                                                    "aux:pass", "case",
                                                    "cc", "cc:preconj",
                                                    "ccomp", "clf",
                                                    "compound", "compound:lvc",
                                                    "compound:prt", "compound:redup",
                                                    "compound:svc", "conj",
                                                    "cop", "csubj",
                                                    "csubj:outer", "csubj:pass",
                                                    "dep", "det",
                                                    "det:numgov", "det:nummod",
                                                    "det:poss", "discourse",
                                                    "dislocated", "expl",
                                                    "expl:impers", "expl:pass",
                                                    "expl:pv", "fixed",
                                                    "flat", "flat:foreign",
                                                    "flat:name", "goeswith",
                                                    "iobj", "list", "mark",
                                                    "nmod", "nmod:poss",
                                                    "nmod:tmod", "nsubj",
                                                    "nsubj:outer", "nsubj:pass",
                                                    "nummod", "nummod:gov",
                                                    "obj", "obl", "obl:agent",
                                                    "obl:arg", "obl:lmod",
                                                    "obl:tmod", "orphan",
                                                    "parataxis", "punct",
                                                    "reparandum", "root",
                                                    "vocative", "xcomp"]},
                "entity": {"type": "span", "on": "tokens", "data": "string" },
        }

    def execute(self, doc):
        """Execute SpaCy on the document"""
        if not hasattr(self, "nlp") or not self.nlp:
            raise Exception("SpaCY model not loaded. "
            + "Please call setup() on the service.")
        result = self.nlp(doc.get_layer("text").raw())
        doc.add_layer("tokens", [
            (w.idx, w.idx + len(w)) for w in result])
        doc.add_layer("pos", [w.pos_ for w in result])
        doc.add_layer("tag", [w.tag_ for w in result])
        doc.add_layer("lemma", [w.lemma_ for w in result])
        doc.add_layer("dep", [(w.head.i, w.dep_) for w in result])
        doc.add_layer("entity", [(e.start, e.end, e.label_) for e in result.ents])


