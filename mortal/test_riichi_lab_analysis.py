import asyncio
import json
import os
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import riichi_lab
from riichi_lab_analysis import TableAnalysis, TenpaiPredictor
from riichi_lab_ui import TableState, score_bar, score_scale
from tenpai_net import TenpaiNet


def opening():
    return {'type': 'start_kyoku', 'bakaze': 'E', 'kyoku': 1, 'honba': 0,
            'kyotaku': 0, 'oya': 0, 'dora_marker': '9m', 'scores': [25000] * 4,
            'tehais': [['?'] * 13, ['1m', '2m', '3m', '1p', '2p', '3p', '1s', '2s',
                                   '3s', 'E', 'E', 'E', '5p'], ['?'] * 13, ['?'] * 13]}


class HandTests(unittest.TestCase):
    def setUp(self):
        self.board = TableState()
        self.analysis = TableAnalysis(self.board)
        self.feed({'type': 'start_game', 'id': 1})
        self.feed(opening())

    def feed(self, event):
        self.board.update(event)
        self.analysis.update(json.dumps(event), event)

    def test_exact_waits_and_furiten(self):
        self.assertEqual(self.board.hand_status, 'TENPAI: 5p')
        self.feed({'type': 'tsumo', 'actor': 1, 'pai': '5p'})
        self.feed({'type': 'dahai', 'actor': 1, 'pai': '5p', 'tsumogiri': True})
        self.assertEqual(self.board.hand_status, 'TENPAI: 5p | FURITEN')

    def test_draw_is_not_mislabeled_as_settled_waits(self):
        self.feed({'type': 'tsumo', 'actor': 1, 'pai': '9s'})
        self.assertIn('after best discard', self.board.hand_status)
        self.assertNotIn('TENPAI:', self.board.hand_status)
        self.feed({'type': 'dahai', 'actor': 1, 'pai': '9s', 'tsumogiri': True})
        self.assertEqual(self.board.hand_status, 'TENPAI: 5p')

    def test_nonready_hand_shows_shanten(self):
        event = opening()
        event['tehais'][1][11] = '9m'
        self.feed(event)
        self.assertEqual(self.board.hand_status, '1 shanten')

    def test_round_reset_invalidates_predictions(self):
        self.board.prediction = {'by_seat': {}}
        before = self.analysis.generation
        self.feed(opening())
        self.assertIsNone(self.board.prediction)
        self.assertGreater(self.analysis.generation, before)
        self.assertEqual(self.board.hand_status, 'TENPAI: 5p')

    def test_failed_analysis_recovers_next_round(self):
        self.feed({'type': 'dahai', 'actor': 1, 'pai': '7m', 'tsumogiri': False})
        self.assertEqual(self.board.hand_status, 'unavailable')
        self.assertFalse(self.analysis.active)
        self.feed(opening())
        self.assertEqual(self.board.hand_status, 'TENPAI: 5p')


class ModelTests(unittest.TestCase):
    def test_training_checkpoint_has_one_sigmoid(self):
        model = TenpaiNet(934, channels=4, blocks=1)
        for parameter in model.parameters():
            parameter.data.zero_()
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, 'tenpai.pth')
            torch.save({'model': model.state_dict()}, filename)
            predictor = TenpaiPredictor.load(filename)
            result = predictor.predict(np.zeros((934, 34), dtype=np.float32))
        for values in result.values():
            self.assertTrue(np.all(np.asarray(values) == 0.5))

    def test_v4_observation_checkpoint_is_rejected(self):
        model = TenpaiNet(1012, channels=4, blocks=1)
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, 'tenpai.pth')
            torch.save({'model': model.state_dict()}, filename)
            with self.assertRaisesRegex(ValueError, 'v3 / 934'):
                TenpaiPredictor.load(filename)

    def test_python_reducers_are_rejected(self):
        class PythonReducer:
            def __reduce__(self):
                return eval, ('1 + 1',)
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, 'tenpai.pth')
            torch.save(PythonReducer(), filename)
            with self.assertRaises(pickle.UnpicklingError):
                TenpaiPredictor.load(filename)

    def test_score_bars_include_high_and_small_positive_scores(self):
        players = [SimpleNamespace(score=s) for s in (17400, 59600, 2100, 20900)]
        ceiling = score_scale(players)
        self.assertEqual(ceiling, 60000)
        self.assertNotEqual(score_bar(2100, ceiling), score_bar(0, ceiling))
        self.assertNotEqual(score_bar(50000, ceiling), score_bar(59600, ceiling))
        self.assertEqual(len(score_bar(59600, ceiling)), 10)


class Socket:
    def __init__(self, events, stay_open=False):
        self.events = iter(events)
        self.closed = False
        self.stay_open = stay_open

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            event = next(self.events)
        except StopIteration:
            if self.stay_open:
                await asyncio.sleep(60)
            raise StopAsyncIteration
        return event if isinstance(event, (bytes, str)) else json.dumps(event)

    async def close(self):
        self.closed = True


def ending_session():
    session = riichi_lab.Session(None)
    session.seat = 1
    session.bot = SimpleNamespace(react=lambda line: None)
    return session


class ReviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_game_limit_waits_for_validation_verdict(self):
        session = ending_session()
        ws = Socket([{'type': 'end_game', 'scores': [25000] * 4},
                     {'type': 'validation_result', 'passed': True}])
        done = await riichi_lab._pump(ws, session, 1, validation=True)
        self.assertTrue(done)
        self.assertTrue(ws.closed)
        self.assertEqual(session.verdicts, [{'type': 'validation_result', 'passed': True}])
        self.assertEqual(session.results, [(2, 25000)])

    async def test_verdict_without_end_game_meets_validation_budget(self):
        session = ending_session()
        ws = Socket([{'type': 'validation_result', 'passed': True}])
        self.assertTrue(await riichi_lab._pump(ws, session, 1, validation=True))
        self.assertEqual(session.games, 0)
        self.assertEqual(session.results, [])

    async def test_missing_verdict_has_bounded_wait(self):
        session = ending_session()
        ws = Socket([{'type': 'end_game', 'scores': [25000] * 4}], stay_open=True)
        self.assertTrue(await riichi_lab._pump(ws, session, 1, validation=True, verdict_timeout=0.01))
        self.assertTrue(ws.closed)
        self.assertEqual(session.verdicts, [])

    async def test_mixed_full_and_verdict_only_runs_share_game_budget(self):
        session = ending_session()
        ws = Socket([{'type': 'end_game', 'scores': [25000] * 4}])
        self.assertFalse(await riichi_lab._pump(ws, session, 2, validation=True))
        session.bot = SimpleNamespace(react=lambda line: None)
        ws = Socket([{'type': 'validation_result', 'passed': True}])
        self.assertTrue(await riichi_lab._pump(ws, session, 2, validation=True))
        self.assertEqual(session.games, 1)
        self.assertEqual(session.validation_cutoffs, 1)

    async def test_nonvalidation_disconnects_at_end_game(self):
        session = ending_session()
        ws = Socket([{'type': 'end_game', 'scores': [25000] * 4}], stay_open=True)
        self.assertFalse(await riichi_lab._pump(ws, session, 0))
        self.assertTrue(ws.closed)

    async def test_ack_and_malformed_frames(self):
        session = ending_session()
        ws = Socket([b'ignored binary', 'null', '[1,2]', 'broken JSON',
                     {'type': 'new_server_event', 'new_field': 1},
                     {'type': 'action_ack', 'request_id': 7, 'status': 'defaulted', 'bank_ms': 0}])
        await riichi_lab._pump(ws, session, 0)
        self.assertEqual(session.last_ack['status'], 'defaulted')
        self.assertEqual(session.time_budget['bank_ms'], 0)

    async def test_legacy_header_keyword_is_chosen_before_await(self):
        captured = {}

        def connect(url, extra_headers=None, **kwargs):
            captured.update(headers=extra_headers, kwargs=kwargs)
            class Connection:
                async def __aenter__(self):
                    self.assert_no_wrong_keyword()
                    return Socket([])
                def assert_no_wrong_keyword(self):
                    assert 'additional_headers' not in kwargs
                async def __aexit__(self, *exc):
                    pass
            return Connection()

        with patch('websockets.connect', connect):
            await riichi_lab.play('ws://127.0.0.1', 'local-test', ending_session(), 0)
        self.assertEqual(captured['headers'], {'Authorization': 'Bearer local-test'})

    async def test_prediction_seat_mapping(self):
        board = TableState()
        analysis = TableAnalysis(board)
        board.seat = 2
        analysis.active = True
        analysis.revision = 1
        analysis.state = SimpleNamespace(encode_obs=lambda version, kan: (np.zeros((934, 34)), None))
        result = {'tenpai': [0.1, 0.2, 0.3], 'furiten': [0.4, 0.5, 0.6],
                  'waits': [[0.1] * 34, [0.2] * 34, [0.3] * 34], 'any_wait': [0.5] * 34}
        analysis.predictor = SimpleNamespace(predict=lambda obs: dict(result))
        worker = asyncio.create_task(analysis.sample())
        try:
            for _ in range(150):
                if board.prediction:
                    break
                await asyncio.sleep(0.01)
            self.assertEqual(board.prediction['by_seat'][3]['tenpai'], 0.1)
            self.assertEqual(board.prediction['by_seat'][0]['tenpai'], 0.2)
            self.assertEqual(board.prediction['by_seat'][1]['tenpai'], 0.3)
            self.assertNotIn(2, board.prediction['by_seat'])
        finally:
            worker.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await worker

    async def test_inflight_prediction_is_discarded_after_pause(self):
        board = TableState()
        board.seat = 1
        analysis = TableAnalysis(board)
        analysis.active = True
        analysis.state = SimpleNamespace(encode_obs=lambda *args: (np.zeros((934, 34)), None))
        analysis.predictor = SimpleNamespace(predict=lambda obs: None)
        started, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def delayed(*args):
            started.set()
            await release.wait()
            finished.set()
            return {}  # Must be discarded before any result fields are read.

        with patch('riichi_lab_analysis.asyncio.to_thread', side_effect=delayed):
            worker = asyncio.create_task(analysis.sample())
            try:
                await asyncio.wait_for(started.wait(), 2)
                analysis.pause()
                release.set()
                await asyncio.wait_for(finished.wait(), 1)
                await asyncio.sleep(0)
                self.assertIsNone(board.prediction)
                self.assertFalse(analysis.active)
                self.assertFalse(worker.done())
            finally:
                worker.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await worker

    def test_fallback_does_not_override_explicit_tedashi(self):
        action, why = riichi_lab.choose(None, [{'type': 'dahai', 'pai': '5p', 'tsumogiri': False}], 1, '5p')
        self.assertFalse(action['tsumogiri'])
        self.assertIsNotNone(why)

    def test_call_keeps_required_fields_without_metadata(self):
        reaction = {'type': 'pon', 'actor': 1, 'target': 2, 'pai': 'E',
                    'consumed': ['E', 'E'], 'meta': {'q_values': [1]}}
        action, why = riichi_lab.choose(reaction, [{'type': 'pon', 'pai': 'E'}], 1)
        self.assertIsNone(why)
        self.assertEqual(action['target'], 2)
        self.assertEqual(action['consumed'], ['E', 'E'])
        self.assertNotIn('meta', action)

    def test_new_event_and_failures_clear_stale_reactions(self):
        session = ending_session()
        session.reaction = {'type': 'dahai', 'pai': '5p'}
        event = {'type': 'end_kyoku'}
        session.on_mjai(json.dumps(event), event)
        self.assertIsNone(session.reaction)
        session.reaction = {'type': 'dahai', 'pai': '5p'}
        session.bot.react = lambda line: (_ for _ in ()).throw(ValueError('bad event'))
        event = {'type': 'dora', 'dora_marker': '1m'}
        session.on_mjai(json.dumps(event), event)
        self.assertIsNone(session.reaction)


if __name__ == '__main__':
    unittest.main()
