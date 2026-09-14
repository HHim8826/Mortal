"""Run with: python -m unittest test_riichi_lab -v (from mortal/)."""
import asyncio
import io
import json
import logging
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console

from riichi_lab_ui import Dashboard, NoticeHandler, TableState, use_dashboard


def start(board, seat=1):
    board.update({'type': 'start_game', 'id': seat,
                  'names': ['Alpha', 'Mortal', 'Beta', 'Gamma']})
    board.update({'type': 'start_kyoku', 'bakaze': 'E', 'kyoku': 1, 'honba': 0,
                  'kyotaku': 0, 'oya': 0, 'dora_marker': '9p',
                  'scores': [25000] * 4,
                  'tehais': [['?'] * 13, ['1m', '2m', '3m', '5m', '5m', '5mr',
                              '2p', '3p', '4p', '7s', '8s', '9s', 'E'],
                             ['?'] * 13, ['?'] * 13]})


def session_view():
    board = TableState()
    start(board)
    return SimpleNamespace(table=board, results=[(2, 31000)], games=1,
                           phase='Playing', phase_since=time.monotonic(),
                           last_received=time.monotonic(), fallbacks=0, slowest=0.02)


class TableTests(unittest.TestCase):
    def setUp(self):
        self.board = TableState()
        start(self.board)

    def test_draw_discard_and_hidden_opponents(self):
        self.board.update({'type': 'tsumo', 'actor': 1, 'pai': '5p'})
        self.assertEqual(self.board.players[1].drawn, '5p')
        self.assertEqual(self.board.wall, 69)
        self.board.update({'type': 'dahai', 'actor': 1, 'pai': '1m', 'tsumogiri': False})
        self.assertEqual(len(self.board.players[1].hand), 13)
        self.assertIn('5p', self.board.players[1].hand)
        self.assertNotIn('1m', self.board.players[1].hand)
        self.assertIsNone(self.board.players[1].drawn)
        self.board.update({'type': 'tsumo', 'actor': 0, 'pai': '?'})
        self.board.update({'type': 'dahai', 'actor': 0, 'pai': 'W'})
        self.assertEqual(self.board.players[0].hand, ['?'] * 13)

    def test_riichi_deposit_and_result_scores(self):
        self.board.update({'type': 'reach', 'actor': 1})
        self.board.update({'type': 'dahai', 'actor': 1, 'pai': 'E'})
        self.assertTrue(self.board.players[1].river[-1].reach)
        for _ in range(2):
            self.board.update({'type': 'reach_accepted', 'actor': 1})
        self.assertEqual(self.board.players[1].score, 24000)
        self.assertEqual(self.board.sticks, 1)
        self.board.update({'type': 'hora', 'actor': 1, 'target': 0,
                           'deltas': [-8000, 9000, 0, 0]})
        self.assertEqual([p.score for p in self.board.players], [17000, 33000, 25000, 25000])
        self.assertEqual(self.board.sticks, 0)
        self.assertIn('ron from seat 0', self.board.result)
        self.board.update({'type': 'end_game', 'scores': [18000, 32000, 25000, 25000]})
        self.assertEqual(self.board.players[1].score, 32000)

    def test_calls_and_added_red_kan(self):
        self.board.update({'type': 'dahai', 'actor': 0, 'pai': '5m'})
        self.board.update({'type': 'pon', 'actor': 1, 'target': 0, 'pai': '5m',
                           'consumed': ['5m', '5m']})
        self.assertTrue(self.board.players[0].river[-1].called)
        self.assertIn('5mr', self.board.players[1].hand)
        self.board.update({'type': 'kakan', 'actor': 1, 'pai': '5mr',
                           'consumed': ['5m'] * 3})
        p = self.board.players[1]
        self.assertNotIn('5mr', p.hand)
        self.assertEqual(p.melds, [['kakan', ['5m', '5m', '5m', '5mr']]])

    def test_closed_and_open_kan(self):
        self.board.players[1].hand = ['5m'] * 3 + ['5mr']
        self.board.update({'type': 'ankan', 'actor': 1, 'consumed': ['5m'] * 3 + ['5mr']})
        self.assertEqual(self.board.players[1].hand, [])
        self.assertEqual(len(self.board.players[1].melds[0][1]), 4)
        self.board.update({'type': 'dora', 'dora_marker': '1m'})
        self.assertEqual(self.board.dora, ['9p', '1m'])
        self.board.update({'type': 'daiminkan', 'actor': 2, 'target': 3, 'pai': '7s',
                           'consumed': ['7s'] * 3})
        self.assertEqual(len(self.board.players[2].hand), 10)

    def test_round_reset_preserves_result_and_names(self):
        self.board.update({'type': 'ryukyoku', 'deltas': [1500, -1500, 1500, -1500]})
        result = self.board.result
        self.board.update({'type': 'start_kyoku', 'bakaze': 'S', 'kyoku': 2,
                           'oya': 1, 'scores': [26500, 23500, 26500, 23500]})
        self.assertEqual(self.board.players[1].name, 'Mortal')
        self.assertEqual(self.board.players[1].river, [])
        self.assertEqual(self.board.result, result)
        self.assertEqual(self.board.wall, 70)
        self.board.update({'type': 'start_game', 'id': 2})
        self.assertEqual(self.board.round, 'Waiting')
        self.assertEqual(self.board.seat, 2)


class DisplayTests(unittest.TestCase):
    def test_terminal_detection_and_plain_override(self):
        stream = SimpleNamespace(isatty=lambda: True)
        with patch.dict(os.environ, {'TERM': 'xterm-256color'}):
            self.assertTrue(use_dashboard('auto', stream))
            self.assertFalse(use_dashboard('plain', stream))
        with patch.dict(os.environ, {'TERM': 'dumb'}):
            self.assertFalse(use_dashboard('auto', stream))
            with self.assertRaises(SystemExit):
                use_dashboard('live', stream)
        self.assertFalse(use_dashboard('auto', io.StringIO()))

    def test_bounded_logs_and_persistent_warning(self):
        notices = NoticeHandler()
        notices.emit(logging.LogRecord('test', logging.WARNING, '', 0, 'Fallback occurred', (), None))
        for i in range(100):
            notices.emit(logging.LogRecord('test', logging.INFO, '', 0, 'Event %d', (i,), None))
        self.assertEqual(len(notices.messages), 4)
        self.assertIn('Fallback occurred', notices.warning)
        self.assertEqual(notices.messages[-1], 'Event 99')

    def test_fallback_notice_does_not_dump_model_metadata(self):
        notices = NoticeHandler()
        notices.emit(logging.LogRecord('test', logging.WARNING, '', 0,
                     'falling back to %s: %s', ('dahai', 'huge q_values metadata'), None))
        self.assertIn('Fallback to dahai', notices.warning)
        self.assertNotIn('q_values', notices.warning)

    def test_render_sizes_and_untrusted_names(self):
        session = session_view()
        session.table.players[2].name = '[blink]Beta[/blink]\x1b\n\t'
        for i in range(24):
            session.table.update({'type': 'dahai', 'actor': i % 4, 'pai': '5mr'})
        for width, height in ((120, 40), (100, 34), (80, 24), (60, 20), (20, 8)):
            with self.subTest(size=(width, height)):
                output = io.StringIO()
                console = Console(file=output, width=width, height=height, color_system=None)
                dashboard = Dashboard(session, 'VALIDATION', 'best_ema', console=console)
                console.print(dashboard.render())
                rendered = output.getvalue()
                self.assertLessEqual(len(rendered.splitlines()), height)
                self.assertTrue(all(len(line) <= width for line in rendered.splitlines()))
                self.assertNotIn('\x1b', rendered)
                if width >= 72:
                    self.assertIn('YOU', rendered)
                    self.assertIn('5mr', rendered)
                    self.assertIn('Ctrl+C', rendered)

    def test_screen_and_log_handlers_restore_on_error(self):
        output = io.StringIO()
        console = Console(file=output, width=80, height=24, force_terminal=True)
        handlers = logging.getLogger().handlers[:]
        with self.assertRaisesRegex(RuntimeError, 'test failure'):
            with Dashboard(session_view(), 'VALIDATION', 'best_ema', console=console) as dashboard:
                logging.warning('captured warning')
                self.assertIn('captured warning', dashboard.notices.warning)
                raise RuntimeError('test failure')
        self.assertEqual(logging.getLogger().handlers, handlers)
        ansi = output.getvalue()
        self.assertIn('\x1b[?1049h', ansi)
        self.assertIn('\x1b[?1049l', ansi)
        self.assertIn('\x1b[?25h', ansi)
        self.assertNotIn('captured warning', ansi)  # never printed above the screen


class ActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from riichi_lab import choose
        cls.choose = staticmethod(choose)

    def test_duplicate_discard_offers_preserve_model_choice(self):
        for tile, count in (('9m', 2), ('7m', 3), ('F', 2)):
            with self.subTest(tile=tile, count=count):
                candidate = {'actor': 0, 'pai': tile, 'type': 'dahai'}
                possible = [dict(candidate) for _ in range(count)]
                possible.append({'type': 'dahai', 'pai': '5p', 'actor': 0})
                reaction = dict(candidate, tsumogiri=False)
                action, why = self.choose(reaction, possible, seat=0, drawn='5p')
                self.assertIsNone(why)
                self.assertEqual(action, candidate)
                self.assertIsNot(action, possible[0])
                self.assertEqual(len(possible), count + 1)

    def test_duplicate_none_and_meld_offers(self):
        for candidate in ({'type': 'none'},
                          {'type': 'ankan', 'actor': 0, 'pai': '5m',
                           'consumed': ['5m', '5m', '5m', '5mr']}):
            with self.subTest(kind=candidate['type']):
                reaction = {k: v for k, v in candidate.items() if k != 'pai'}
                if 'consumed' in reaction:
                    reaction['consumed'] = list(reversed(reaction['consumed']))
                action, why = self.choose(reaction, [candidate, dict(reversed(list(candidate.items())))], 0)
                self.assertIsNone(why)
                self.assertEqual(action, dict(candidate, actor=0))

    def test_distinct_server_fields_still_require_fallback(self):
        reaction = {'type': 'dahai', 'pai': '9m'}
        candidates = [dict(reaction, tsumogiri=False), dict(reaction, tsumogiri=True)]
        possible = candidates + [dict(candidates[0]), {'type': 'none'}]
        action, why = self.choose(reaction, possible, 0)
        self.assertEqual(action, {'type': 'none', 'actor': 0})
        self.assertIn('2 offers fit', why)

    def test_red_five_and_absent_reaction_keep_fallback(self):
        possible = [{'type': 'dahai', 'pai': '5m'}] * 2 + [{'type': 'none'}]
        for reaction in (None, {'type': 'dahai', 'pai': '5mr'}):
            action, why = self.choose(reaction, possible, 0)
            self.assertEqual(action, {'type': 'none', 'actor': 0})
            self.assertIsNotNone(why)


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_buffered_events_allow_ui_to_run(self):
        import riichi_lab
        ticks = []

        class BufferedSocket:
            def __aiter__(self):
                return self

            async def __anext__(self):
                if len(ticks) == 100:
                    raise StopAsyncIteration
                ticks.append('event')
                return '{"type":"action_ack"}'

        async def observe():
            await asyncio.sleep(0)
            self.assertLess(len(ticks), 100)

        await asyncio.gather(riichi_lab._pump(BufferedSocket(), riichi_lab.Session(None), 0),
                             observe())

    async def test_reconnect_status_and_once(self):
        import riichi_lab
        session = riichi_lab.Session(None)
        args = SimpleNamespace(url='ws://127.0.0.1', token='local-test', games=0,
                               once=True, log_dir=None)

        async def disconnected(*args):
            session.analysis.active = True
            session.table.prediction = {'captured': 0}
            raise ConnectionError('local test disconnect')

        with patch.object(riichi_lab, 'play', side_effect=disconnected):
            result = await riichi_lab.run(args, None, session)
        self.assertIs(result, session)
        self.assertEqual(session.phase, 'Disconnected')
        self.assertFalse(session.analysis.active)
        self.assertIsNone(session.table.prediction)

    async def test_local_websocket_keeps_actions_and_state(self):
        import riichi_lab
        from websockets.asyncio.server import serve

        events = [
            {'type': 'start_game', 'id': 1},
            {'type': 'start_kyoku', 'bakaze': 'E', 'kyoku': 1, 'honba': 0,
             'kyotaku': 0, 'oya': 0, 'dora_marker': '1m', 'scores': [25000] * 4,
             'tehais': [['?'] * 13, ['1m', '2m', '3m', '1p', '2p', '3p', '1s', '2s',
                                    '3s', 'E', 'E', 'E', '5p'], ['?'] * 13, ['?'] * 13]},
            {'type': 'tsumo', 'actor': 1, 'pai': '5mr'},
            {'type': 'request_action', 'request_id': 'turn-123',
             'possible_actions': [{'type': 'dahai', 'pai': '5mr'},
                                  {'type': 'dahai', 'pai': '5mr'}]},
            {'type': 'dahai', 'actor': 1, 'pai': '5mr', 'tsumogiri': True},
            {'type': 'ryukyoku', 'deltas': [-1000, 3000, -1000, -1000]},
            {'type': 'end_game', 'scores': [24000, 28000, 24000, 24000]},
            {'type': 'validation_result', 'passed': True},
        ]
        received = []

        async def server(ws):
            for event in events:
                await ws.send(json.dumps(event))
                if event['type'] == 'request_action':
                    received.append(json.loads(await ws.recv()))

        fake_bot = SimpleNamespace(react=lambda line: json.dumps({'type': 'dahai', 'pai': '5mr'})
                                   if json.loads(line)['type'] == 'tsumo' else None)
        with patch('libriichi.mjai.Bot', return_value=fake_bot):
            async with serve(server, '127.0.0.1', 0) as host:
                port = host.sockets[0].getsockname()[1]
                session = riichi_lab.Session(None)
                await asyncio.wait_for(riichi_lab.play(f'ws://127.0.0.1:{port}/ws/validate', 'local-test',
                                                       session, games=0), timeout=5)
        self.assertEqual(received, [{'type': 'dahai', 'pai': '5mr', 'actor': 1,
                                     'request_id': 'turn-123'}])
        self.assertEqual(session.results, [(1, 28000)])
        self.assertEqual(session.table.players[1].score, 28000)
        self.assertEqual(session.fallbacks, 0)
        self.assertEqual(session.verdicts, [{'type': 'validation_result', 'passed': True}])
        self.assertIsNotNone(session.last_received)

    async def test_render_sampling_and_cleanup_on_cancellation(self):
        session = session_view()
        console = Console(file=io.StringIO(), width=80, height=24, force_terminal=True)
        with Dashboard(session, 'VALIDATION', 'best_ema', refresh=2, console=console) as dashboard:
            async def burst():
                for _ in range(500):
                    session.table.update({'type': 'tsumo', 'actor': 0, 'pai': '?'})
                await asyncio.sleep(0.05)
                raise asyncio.CancelledError

            with patch.object(dashboard.live, 'update', wraps=dashboard.live.update) as update:
                with self.assertRaises(asyncio.CancelledError):
                    await dashboard.run(burst())
                self.assertLessEqual(update.call_count, 2)
        self.assertEqual(len(asyncio.all_tasks()), 1)


if __name__ == '__main__':
    unittest.main()
