import torch
from torch import nn
from torch.nn import functional as F


class CaptchaModel(nn.Module):
    """https://github.com/ritikjain51/CaptchaExtractor-Pytorch"""
    def __init__(self, num_chars):
        super(CaptchaModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.linear_1 = nn.Linear(4608, 64)
        self.dropout = nn.Dropout(0.3)

        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.24)
        self.output = nn.Linear(64, num_chars + 1)

    def forward(self, images, target=None):
        bs, c, h, w = images.shape
        # print(bs, c, h, w)
        x = F.relu(self.conv1(images))
        # print(x.size())
        x = self.max_pool1(x)
        # print(x.size())

        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        # print(x.size())
        x = x.permute(0, 3, 1, 2)
        # print(x.size())

        x = x.view(bs, x.size(1), -1)
        # print(x.size())

        x = self.linear_1(x)
        x = self.dropout(x)

        x, _ = self.gru(x)
        # print(x.size())
        x = F.softmax(self.output(x))
        # print(x.size())
        x = x.permute(1, 0, 2)

        if target is not None:
            log_softmax = F.log_softmax(x, 2)
            input_length = torch.full(
                size=(bs,),
                fill_value=log_softmax.size(2),
                dtype=torch.int32
            )
            print(input_length)
            target_length = torch.full(
                size=(bs,),
                fill_value=target.size(1),
                dtype=torch.int32
            )
            loss = nn.CTCLoss()(
                log_softmax,
                target,
                input_length, target_length
            )
            return x, loss
        # we are here if target is None
        return x, None