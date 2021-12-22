import torch
import time

from transformers import AutoTokenizer, AutoModelForQuestionAnswering


class DocumentReader:
    def __init__(self, pretrained_model_name_or_path='bert-large-uncased'):
        self.READER_PATH = pretrained_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.READER_PATH)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.READER_PATH)
        self.max_len = self.model.config.max_position_embeddings
        self.chunked = False

    def tokenize(self, question, text):
        try:
            self.inputs = self.tokenizer.encode_plus(
                question, text, add_special_tokens=True, return_tensors="pt")
            self.input_ids = self.inputs["input_ids"].tolist()[0]
        except:
            # if len(self.input_ids) > self.max_len:
            self.inputs = self.chunkify()
            self.chunked = True

    def chunkify(self):
        """ 
        Break up a long article into chunks that fit within the max token
        requirement for that Transformer model. 

        Calls to BERT / RoBERTa / ALBERT require the following format:
        [CLS] question tokens [SEP] context tokens [SEP].
        """
        print("CHUNKING RUNNING")
        # create question mask based on token_type_ids
        # value is 0 for question tokens, 1 for context tokens
        qmask = self.inputs['token_type_ids'].lt(1)
        qt = torch.masked_select(self.inputs['input_ids'], qmask)
        chunk_size = self.max_len - qt.size()[0] - 1  # the "-1" accounts for
        # having to add an ending [SEP] token to the end

        # create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
        chunked_input = OrderedDict()
        for k, v in self.inputs.items():
            q = torch.masked_select(v, qmask)
            c = torch.masked_select(v, ~qmask)
            chunks = torch.split(c, chunk_size)

            for i, chunk in enumerate(chunks):
                if i not in chunked_input:
                    chunked_input[i] = {}

                thing = torch.cat((q, chunk))
                if i != len(chunks)-1:
                    if k == 'input_ids':
                        thing = torch.cat((thing, torch.tensor([102])))
                    else:
                        thing = torch.cat((thing, torch.tensor([1])))

                chunked_input[i][k] = torch.unsqueeze(thing, dim=0)
        return chunked_input

    def get_answer(self):
        if self.chunked:
            answer = ''
            for k, chunk in self.inputs.items():
                answer_start_scores, answer_end_scores = self.model(
                    **chunk, return_dict=False)

                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1

                ans = self.convert_ids_to_string(
                    chunk['input_ids'][0][answer_start:answer_end])
                if ans != '[CLS]':
                    answer += ans + " / "
            return answer
        else:
            answer_start_scores, answer_end_scores = self.model(
                **self.inputs, return_dict=False)

            # get the most likely beginning of answer with the argmax of the score
            answer_start = torch.argmax(answer_start_scores)
            # get the most likely end of answer with the argmax of the score
            answer_end = torch.argmax(answer_end_scores) + 1

            return self.convert_ids_to_string(self.inputs['input_ids'][0][
                                              answer_start:answer_end])

    def convert_ids_to_string(self, input_ids):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids))


# EXAMPLE RUN

model_path_longformer = "mrm8488/longformer-base-4096-finetuned-squadv2"

reader = DocumentReader(model_path_longformer)


question = "Does it work on S21?"
context = "***update****Just an update on this a few months later... this particular case seems to, for want of better terms, . On the right side where the volume and power buttons are... the case seems to have gotten stretched a bit and has a bulge and noticeable gap between the screen and the case in this spot. I suspect it's because the plastic shield that fits over the phone is anchoring the rubber just below this point and use of the buttons above that anchor point have stretched the rubber. Not entirety sure it's affecting protection of the phone but it is a little annoying because the rubber buttons no longer perfectly live up with their counterparts on the device. \
First time commuter user. Had defender for my S7 and Symmetry for my S9. Ordered this one for the S21 because i was intrigued by the blue color option. Every bit as good as every other otterbox line. You shouldn't be disappointed unless you run it over with a tank... Side note to amazon....the star ratings for each category should have an option for . The readers on many new phones, including mine, are built into the screen of the phone and have nothing to do with and are not affected by the case\
\I wanted a less rugged case that would also protect my new phone from my clumsiness. I've dropped it back-down a few times with no issue, but I would be careful dropping it face-down as the lip isn't very prominent. It's easy to disassemble to clean. My only issue is after a few months the silicone portion with the buttons refuses to stay flush to my phone which is a bit annoying.\
It's a rather smooth case and I have pretty dry hands so I feel like I'm going to drop it pretty often, but it feels durable so far. My old phone had an Otterbox Defender case on it and I liked that one much better.Ordered this one for the S21 because i was intrigued by the blue color option. Every bit as good as every other otterbox line. You shouldn't be disappointed unless you run it over with a tank... Side note to amazon....the star ratings for each category should have an option for . The readers on many new phones, including mine, are built into the screen of the phone and have nothing to do with and are not affected by the case\
\I wanted a less rugged case that would also protect my new phone from my clumsiness. I've dropped it back-down a few times with no issue, but I would be careful dropping it face-down as the lip isn't very prominent. It's easy to disassemble to clean. My only issue is after a few months the silicone portion with the buttons refuses to stay flush to my phone which is a bit annoying.\
It's a rather smooth case and I have pretty dry hands so I feel like I'm going to drop it pretty often, but it feels durable so far. My old phone had an Otterbox Defender case on it and I liked that one much better."

start = time.time()
reader.tokenize(question, context)
print(f"Answer: {reader.get_answer()}")
end = time.time()
print(end - start)


def question_answer(question, reviews_text):

    return "Longformer Answer"
