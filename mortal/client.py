import prelude

import logging
import os
import socket
import torch
import numpy as np
import time
import gc
from os import path
from torch.utils.tensorboard import SummaryWriter
from model import Brain, head_for
from player import TrainPlayer
from common import send_msg, recv_msg
from config import config

def main():
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    device = torch.device(config['control']['device'])
    version = config['control']['version']
    num_blocks = config['resnet']['num_blocks']
    conv_channels = config['resnet']['conv_channels']

    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels).to(device).eval()
    # A policy-gradient run publishes a policy head in the same slot, and the
    # engine plays it the same way. The trainer decides which; the workers are
    # told by the config they share with it.
    head_kind = config['online'].get('head', 'dqn')
    logging.info(f'playing the {head_kind} head')
    dqn = head_for(head_kind, version=version).to(device)
    if config['online']['enable_compile']:
        mortal.compile()
        dqn.compile()

    train_player = TrainPlayer()
    param_version = -1

    # The trainee's average against the frozen opponent, session by session.
    # It is the first number that moves when a policy improves or breaks -- it
    # caught the v4 degradation hours before the 10,000-step evaluations -- and
    # until now it existed only as a line in each worker's log. Each worker
    # writes its own series; a run with several of them reads as one cloud.
    worker = os.environ.get('MORTAL_WORKER', '0')
    # The launcher points this at the run being played, so two variants' workers
    # never write into one series.
    tb_dir = os.environ.get('MORTAL_TB_DIR') or config['control']['tensorboard_dir']
    writer = SummaryWriter(path.join(tb_dir, 'selfplay', f'worker{worker}'))
    session = 0

    pts = np.array([90, 45, 0, -135])
    history_window = config['online']['history_window']
    history = []

    while True:
        while True:
            with socket.socket() as conn:
                conn.connect(remote)
                msg = {
                    'type': 'get_param',
                    'param_version': param_version,
                }
                send_msg(conn, msg)
                rsp = recv_msg(conn, map_location=device)
                if rsp['status'] == 'ok':
                    param_version = rsp['param_version']
                    break
                time.sleep(3)
        mortal.load_state_dict(rsp['mortal'])
        dqn.load_state_dict(rsp['dqn'])
        logging.info('param has been updated')

        started = time.time()
        rankings, file_list = train_player.train_play(mortal, dqn, device)
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ pts / rankings.sum()

        history.append(np.array(rankings))
        if len(history) > history_window:
            del history[0]
        sum_rankings = np.sum(history, axis=0)
        ma_avg_rank = sum_rankings @ np.arange(1, 5) / sum_rankings.sum()
        ma_avg_pt = sum_rankings @ pts / sum_rankings.sum()

        logging.info(f'trainee rankings: {rankings} ({avg_rank:.6}, {avg_pt:.6}pt)')
        logging.info(f'last {len(history)} sessions: {sum_rankings} ({ma_avg_rank:.6}, {ma_avg_pt:.6}pt)')

        session += 1
        writer.add_scalar('selfplay/avg_rank', avg_rank, session)
        writer.add_scalar('selfplay/avg_pt', avg_pt, session)
        writer.add_scalar('selfplay/avg_rank_ma', ma_avg_rank, session)
        writer.add_scalar('selfplay/avg_pt_ma', ma_avg_pt, session)
        writer.add_scalar('selfplay/param_version', param_version, session)
        writer.add_scalar('selfplay/hanchans_per_second',
                          rankings.sum() / max(time.time() - started, 1e-9), session)
        writer.flush()

        logs = {}
        for filename in file_list:
            with open(filename, 'rb') as f:
                logs[path.basename(filename)] = f.read()

        with socket.socket() as conn:
            conn.connect(remote)
            send_msg(conn, {
                'type': 'submit_replay',
                'logs': logs,
                'param_version': param_version,
            })
            logging.info('logs have been submitted')
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
