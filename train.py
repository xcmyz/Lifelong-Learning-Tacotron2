import torch
import torch.nn as nn
from torch import optim

from network import Tacotron2
from data_utils import DataLoader, collate_fn
from data_utils import Tacotron2Dataset
from data_utils import get_old_dataset
from loss_function import Tacotron2Loss
import hparams as hp
from ewc import EWC

from multiprocessing import cpu_count
import numpy as np
import argparse
import os
import time


def main(args):
    # Get device
    device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')

    # Define model
    model = nn.DataParallel(Tacotron2(hp)).to(device)
    ewc = None
    print("Model Have Been Defined")
    num_param = sum(param.numel() for param in model.parameters())
    print('Number of Tacotron Parameters:', num_param)

    # Get dataset
    dataset = Tacotron2Dataset(hp.dataset_path)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay)

    # Criterion
    criterion = Tacotron2Loss()

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step %d---\n" % args.restore_step)
        if args.ewc_train:
            old_dataset = get_old_dataset(hp.dataset_path, hp.sample_num)
            ewc = EWC(model, old_dataset)

    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)

    # Init logger
    if not os.path.exists(hp.logger_path):
        os.mkdir(hp.logger_path)

    # Define Some Information
    Time = np.array([])
    Start = time.clock()

    # Training
    model = model.train()

    for epoch in range(hp.epochs):
        # Get training loader
        training_loader = DataLoader(dataset,
                                     batch_size=hp.batch_size**2,
                                     shuffle=True,
                                     collate_fn=collate_fn,
                                     drop_last=True,
                                     num_workers=cpu_count())
        total_step = hp.epochs * len(training_loader) * hp.batch_size

        for i, batchs in enumerate(training_loader):
            for j, data_of_batch in enumerate(batchs):
                start_time = time.clock()

                current_step = i * hp.batch_size + j + args.restore_step + \
                    epoch * len(training_loader)*hp.batch_size + 1

                # Init
                optimizer.zero_grad()

                # Get Data
                character = torch.from_numpy(
                    data_of_batch["text"]).long().to(device)
                mel_target = torch.from_numpy(data_of_batch["mel_target"]).float().to(
                    device).contiguous().transpose(1, 2)
                stop_target = torch.from_numpy(
                    data_of_batch["stop_token"]).float().to(device)
                input_lengths = torch.from_numpy(
                    data_of_batch["length_text"]).int().to(device)
                output_lengths = torch.from_numpy(
                    data_of_batch["length_mel"]).int().to(device)
                speaker_id = torch.Tensor([0 for _ in range(hp.speaker_size)])
                speaker_id[args.speaker_id] = 1
                speaker_id = speaker_id.float().to(device)

                # Forward
                batch = character, input_lengths, mel_target, stop_target, output_lengths, speaker_id
                x, y = model.module.parse_batch(batch)

                y_pred = model(x)

                # Cal Loss
                mel_loss, mel_postnet_loss, stop_pred_loss = criterion(
                    y_pred, y)
                total_loss = mel_loss + mel_postnet_loss + stop_pred_loss

                ewc_loss = None
                if args.ewc_train:
                    ewc_loss = args.importance * ewc.penalty(model)
                    total_loss += ewc_loss

                # Logger
                t_l = total_loss.item()
                m_l = mel_loss.item()
                m_p_l = mel_postnet_loss.item()
                s_l = stop_pred_loss.item()

                with open(os.path.join("logger", "total_loss.txt"), "a") as f_total_loss:
                    f_total_loss.write(str(t_l)+"\n")

                with open(os.path.join("logger", "mel_loss.txt"), "a") as f_mel_loss:
                    f_mel_loss.write(str(m_l)+"\n")

                with open(os.path.join("logger", "mel_postnet_loss.txt"), "a") as f_mel_postnet_loss:
                    f_mel_postnet_loss.write(str(m_p_l)+"\n")

                with open(os.path.join("logger", "stop_pred_loss.txt"), "a") as f_s_loss:
                    f_s_loss.write(str(s_l)+"\n")

                if args.ewc_train:
                    with open(os.path.join("logger", "ewc_loss.txt"), "a") as f_e_loss:
                        f_e_loss.write(str(ewc_loss.item())+"\n")

                # Backward
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(model.parameters(), 1.)

                # Update weights
                optimizer.step()
                adjust_learning_rate(optimizer, current_step)

                # Print
                if current_step % hp.log_step == 0:
                    Now = time.clock()

                    str1 = "Epoch [{}/{}], Step [{}/{}], Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f};".format(
                        epoch+1, hp.epochs, current_step, total_step, mel_loss.item(), mel_postnet_loss.item())
                    str2 = "Stop Predicted Loss: {:.4f}, Total Loss: {:.4f}.".format(
                        stop_pred_loss.item(), total_loss.item())
                    str3 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                        (Now-Start), (total_step-current_step)*np.mean(Time))

                    print("\n" + str1)
                    print(str2)
                    if args.ewc_train:
                        print("*EWC Loss is {:.6f}*".format(ewc_loss.item()))
                    print(str3)

                    with open(os.path.join("logger", "logger.txt"), "a") as f_logger:
                        f_logger.write(str1 + "\n")
                        f_logger.write(str2 + "\n")
                        f_logger.write(str3 + "\n")
                        f_logger.write("\n")

                if current_step % hp.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                    print("save model at step %d ..." % current_step)

                end_time = time.clock()
                Time = np.append(Time, end_time - start_time)
                if len(Time) == hp.clear_Time:
                    temp_value = np.mean(Time)
                    Time = np.delete(
                        Time, [i for i in range(len(Time))], axis=None)
                    Time = np.append(Time, temp_value)


def adjust_learning_rate(optimizer, step):
    if step == 500000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    elif step == 1000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003

    elif step == 2000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--speaker_id', type=int, default=0)
    parser.add_argument('--ewc_train', type=bool, default=False)
    parser.add_argument('---importance', type=float, default=1000.0)
    args = parser.parse_args()

    main(args)
