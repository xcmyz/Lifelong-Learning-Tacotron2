from torch import nn


class Tacotron2Loss(nn.Module):
    """Tacotron2 Loss"""

    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False

        mel_out, mel_out_postnet, gate_out, _ = model_output
        # print(mel_out.size())

        mel_loss = nn.MSELoss()(mel_out, mel_target)
        mel_postnet_loss = nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        return mel_loss, mel_postnet_loss, gate_loss
