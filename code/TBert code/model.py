import torch.nn as nn


class BTModel(nn.Module):
    def __init__(self, codeEncoder, code_hidden_size, num_class):
        super(BTModel, self).__init__()
        self.codeEncoder = codeEncoder
        self.code_hidden_size = code_hidden_size
        self.num_class = num_class
        for param in self.codeEncoder.parameters():
            param.requires_grad = True

    def forward(self, text_input_ids=None, labels=None):
        # code_output = self.codeEncoder(code_input_ids, attention_mask=code_input_ids.ne(1))
        code_output = self.codeEncoder(input_ids=text_input_ids, attention_mask=text_input_ids.ne(1), labels=labels)

        if labels is not None:
            loss = code_output[0]
            prob = code_output[1]
            return loss, prob
        else:
            prob = code_output[0]
            return prob
