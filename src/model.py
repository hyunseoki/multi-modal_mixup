import torch
import torch.nn as nn


class DaconLSTM(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=200, num_layers=1, output_dim=25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_dim = input_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 75)
        self.fc2 = nn.Linear(75, output_dim)

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.fc2(out)

        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)

        return h0, c0


class DaconModel(nn.Module):
    def __init__(self, model_cnn, model_rnn, num_classes=25):
        super().__init__()
        self.model_cnn = model_cnn
        self.model_rnn = model_rnn
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

        self.classifier = nn.Linear(num_classes * 2, num_classes)

    def forward(self, img, csv):
        out_cnn = self.model_cnn(img)
        out_cnn = self.dropout1(out_cnn)
        out_rnn = self.model_rnn(csv)
        out_rnn = self.dropout2(out_rnn)
        out = torch.cat((out_cnn, out_rnn), dim=1)
        out = self.classifier(out)

        return out


if __name__ == '__main__':
    model = DaconLSTM()
    x = torch.randn(size=(1, 400, 9))
    print(model(x).shape)