import torch
import torch.nn as nn


class ADVModel(nn.Module):
    def __init__(self):
        super(ADVModel, self).__init__()
        # 初始化模型
        input_size = 50
        hidden_size = 50
        self.issue_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.commit_text_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.commit_code_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, issue_text, commit_text, commit_code, labels):
        issue_features = self.issue_lstm(issue_text.unsqueeze(1))[1][0][-1]
        commit_text_features = self.commit_text_lstm(commit_text.unsqueeze(1))[1][0][-1]
        commit_code_features = self.commit_code_lstm(commit_code.unsqueeze(1))[1][0][-1]

        cos_sim_text = torch.cosine_similarity(issue_features, commit_text_features)
        cos_sim_code = torch.cosine_similarity(issue_features, commit_code_features)

        cs = torch.max(cos_sim_text, cos_sim_code)
        logits = torch.stack((1 - cs, cs), dim=1)
        loss = torch.mean(torch.abs(labels - cs))
        return loss, logits
