import torch
from transformers import BloomForCausalLM, AutoConfig, AutoTokenizer

class Generator:
    def __init__(self) -> None:
        model_path = "/mnt/sources/zalo_test/BLOOM-fine-tuning/output/checkpoint-20"
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = BloomForCausalLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
        self.config = AutoConfig.from_pretrained(model_path)
        self.generate_kwargs = dict(max_new_tokens=512, do_sample=False)

        self.model.eval()

    def infer(self, texts, max_seq_length=512):
        input_tokens = self.tokenizer.batch_encode_plus(
                texts,
                return_tensors="pt",
                max_length=max_seq_length,
                padding="max_length",
                truncation=True,
            )
        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.device)
        outputs = self.model.generate(**input_tokens, **self.generate_kwargs)

        input_tokens_lengths = [x.shape[0] for x in input_tokens.input_ids]
        output_tokens_lengths = [x.shape[0] for x in outputs]

        total_new_tokens = [o - i for i, o in zip(input_tokens_lengths, output_tokens_lengths)]
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return texts, outputs, total_new_tokens


if __name__ == '__main__':
    a = Generator()
    print (a.infer(["hello Dung Doan"]))