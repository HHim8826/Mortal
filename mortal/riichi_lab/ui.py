"""Quiet, fixed-position RiichiLab dashboard; no inference or network work here.

Rich's alternate screen keeps updates out of scrollback. Rendering is sampled
by the asyncio loop, not triggered by every tile, so AI play never queues UI
frames. ASCII tile names avoid font-dependent mahjong emoji widths.
"""
import asyncio
import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field


WINDS = ('E', 'S', 'W', 'N')


def clean(value):
    """Server text is literal content, never terminal controls or Rich markup."""
    return ''.join(c for c in str(value) if c.isprintable())


def tile_key(tile):
    if len(tile) >= 2 and tile[0].isdigit() and tile[1] in 'mps':
        return ('mps'.index(tile[1]), int(tile[0]), tile.endswith('r'))
    return (3, 'ESWNPFC?'.find(tile), False)


def score_scale(players):
    highest = max(p.score for p in players)
    return max(50000, ((highest + 9999) // 10000) * 10000)


def score_bar(score, ceiling, width=10):
    """Eighth-cell resolution keeps small positive scores visible."""
    units = max(1, round(max(0, score) / ceiling * width * 8)) if score > 0 else 0
    full, fraction = divmod(min(width * 8, units), 8)
    filled = '█' * full + ('▏▎▍▌▋▊▉'[fraction - 1] if fraction else '')
    return filled + '─' * (width - len(filled))


@dataclass
class Discard:
    tile: str
    drawn: bool = False
    reach: bool = False
    called: bool = False


@dataclass
class Player:
    name: str
    score: int = 25000
    hand: list = field(default_factory=list)
    drawn: str | None = None
    river: list = field(default_factory=list)
    melds: list = field(default_factory=list)
    reach: bool = False
    accepted: bool = False
    delta: int = 0


class TableState:
    """A spectator view of mjai, independent of Mortal's decision state."""

    def __init__(self):
        self.players = [Player(f'Seat {i}') for i in range(4)]
        self.seat = None
        self.round = 'Waiting'
        self.dealer = None
        self.honba = self.sticks = 0
        self.wall = None
        self.dora = []
        self.result = 'No completed hand yet'
        self.last_move = 'Waiting for table events'
        self.hand_status = 'waiting'
        self.prediction = None
        self.prediction_status = 'Opponent estimates off'

    def _remove(self, player, tile):
        if tile in player.hand:
            player.hand.remove(tile)
        elif '?' in player.hand:
            player.hand.remove('?')

    def update(self, event):
        kind = event['type']
        actor = event.get('actor')
        player = self.players[actor] if actor in range(4) else None
        tile = event.get('pai', '?')
        if kind == 'start_game':
            prediction_status = self.prediction_status
            self.__init__()
            self.prediction_status = prediction_status
            self.seat = event.get('id', 0)
            for p, name in zip(self.players, event.get('names') or []):
                p.name = clean(name)
        elif kind == 'start_kyoku':
            names = [p.name for p in self.players]
            self.players = [Player(n) for n in names]
            self.round = f'{event.get("bakaze", "?")}{event.get("kyoku", "?")}'
            self.dealer = event.get('oya', 0)
            self.honba = event.get('honba', 0)
            self.sticks = event.get('kyotaku', 0)
            self.wall = 70
            self.dora = [event.get('dora_marker', '?')]
            self.last_move = 'Hand started'
            for p, score in zip(self.players, event.get('scores') or []):
                p.score = score
            for p, hand in zip(self.players, event.get('tehais') or []):
                p.hand = list(hand)
        elif kind == 'tsumo' and player:
            player.hand.append(tile)
            player.drawn = tile
            if self.wall is not None:
                self.wall = max(0, self.wall - 1)
        elif kind == 'dahai' and player:
            self._remove(player, tile)
            player.drawn = None
            player.river.append(Discard(tile, event.get('tsumogiri', False),
                                        player.reach and not player.accepted))
        elif kind in ('chi', 'pon', 'daiminkan', 'ankan', 'kakan', 'kan') and player:
            consumed = list(event.get('consumed') or [])
            if kind == 'kakan':
                self._remove(player, tile)
                for meld in player.melds:
                    if meld[0] == 'pon' and any(t.rstrip('r') == tile.rstrip('r')
                                               for t in meld[1]):
                        meld[0] = 'kakan'
                        meld[1].append(tile)
                        break
                else:
                    player.melds.append([kind, consumed + [tile]])
            else:
                for used in consumed:
                    self._remove(player, used)
                called = kind in ('chi', 'pon', 'daiminkan')
                player.melds.append([kind, consumed + ([tile] if called else [])])
                target = event.get('target')
                if called and target in range(4) and self.players[target].river:
                    self.players[target].river[-1].called = True
            player.drawn = None
        elif kind == 'reach' and player:
            player.reach = True
        elif kind == 'reach_accepted' and player and not player.accepted:
            player.reach = player.accepted = True
            player.score -= 1000
            self.sticks += 1
        elif kind == 'dora':
            self.dora.append(event.get('dora_marker', '?'))
        elif kind in ('hora', 'ryukyoku'):
            deltas = event.get('deltas') or [0, 0, 0, 0]
            for p, delta in zip(self.players, deltas):
                p.delta += delta
                p.score += delta
            if kind == 'hora':
                target = event.get('target')
                win = 'tsumo' if actor == target else f'ron from seat {target}'
                result = f'Seat {actor} {win}'
                self.sticks = 0
            else:
                result = clean(event.get('reason', 'Exhaustive / abortive draw'))
            self.result = f'{self.round}-{self.honba}: {result} | ' + ' / '.join(
                f'{d:+,}' for d in deltas)
        if kind in ('hora', 'ryukyoku', 'end_game') and event.get('scores'):
            for p, score in zip(self.players, event['scores']):
                p.score = score
        if player and kind in ('dahai', 'chi', 'pon', 'daiminkan', 'ankan', 'kakan', 'reach'):
            self.last_move = f'Seat {actor}  {kind}  {tile if kind == "dahai" else ""}'


class NoticeHandler(logging.Handler):
    """Bounded log area; warnings stay visible until a newer warning arrives."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.messages = deque(maxlen=4)
        self.warning = ''

    def emit(self, record):
        message = clean(record.getMessage())
        if record.msg == 'falling back to %s: %s':
            message = f'Fallback to {clean(record.args[0])}: model choice has no unique legal match.'
        self.messages.append(message)
        if record.levelno >= logging.WARNING:
            self.warning = f'{record.levelname}: {message}'


def use_dashboard(mode, stream=None):
    stream = sys.stderr if stream is None else stream
    capable = stream.isatty() and os.environ.get('TERM') != 'dumb'
    if mode == 'live' and not capable:
        raise SystemExit('--display live needs an interactive terminal (stderr)')
    return mode == 'live' or (mode == 'auto' and capable)


class Dashboard:
    def __init__(self, session, mode, model, refresh=2, console=None):
        try:
            from rich.console import Console
        except ImportError as exc:
            raise SystemExit('Live display needs Rich: pip install rich '
                             '(or use --display plain)') from exc
        self.console = console or Console(stderr=True)
        self.session = session
        self.mode = mode
        self.model = model
        self.refresh = refresh
        self.notices = NoticeHandler()
        self.live = None

    def __enter__(self):
        from rich.live import Live
        root = logging.getLogger()
        # prelude configures a StreamHandler on import. Merely adding a Rich
        # handler would leave that handler scrolling behind the dashboard.
        self.handlers = root.handlers[:]
        root.handlers = [self.notices] + [h for h in self.handlers
                                        if isinstance(h, logging.FileHandler)]
        self.live = Live(self.render(), console=self.console, screen=True,
                         auto_refresh=False, vertical_overflow='crop')
        try:
            self.live.start(refresh=True)
        except BaseException:
            root.handlers = self.handlers
            self.live.stop()
            raise
        return self

    def __exit__(self, *exc):
        try:
            self.live.stop()
        finally:
            logging.getLogger().handlers = self.handlers

    async def run(self, awaitable):
        async def refresh():
            while True:
                self.live.update(self.render(), refresh=True)
                await asyncio.sleep(1 / self.refresh)

        pulse = asyncio.create_task(refresh())
        try:
            return await awaitable
        finally:
            pulse.cancel()
            try:
                await pulse
            except asyncio.CancelledError:
                pass

    def render(self):
        from rich.console import Group
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        session, board = self.session, self.session.table
        width, height = self.console.size
        ranks = [p for p, _ in session.results]
        average = f'{sum(ranks) / len(ranks):.3f}' if ranks else '--'
        age = int(max(0, time.monotonic() - session.phase_since))
        heard = ('--' if session.last_received is None else
                 f'{int(max(0, time.monotonic() - session.last_received))}s')
        status = f'{session.phase} {age // 60:02}:{age % 60:02} | RX {heard}'
        summary = (f'Games {session.games} | Avg {average} | '
                   f'Fallbacks {session.fallbacks} | Max {session.slowest * 1000:.0f}ms')
        prediction = board.prediction
        predicted_seats = prediction['by_seat'] if prediction else {}
        prediction_note = (f'Opponent estimates {int(time.monotonic() - prediction["captured"])}s old; '
                           'waits are not ron odds' if prediction else board.prediction_status)

        def text(value, style=''):
            return Text(clean(value), style=style, no_wrap=True, overflow='ellipsis')

        def panel(content, title=None):
            return Panel(content, title=text(title, 'dim') if title else None,
                         border_style='bright_black', padding=(0, 1))

        def tiles(values):
            line = Text(no_wrap=True, overflow='ellipsis')
            for tile in values:
                style = ('yellow' if tile.endswith('r') else
                         {'m': 'cyan', 'p': 'blue', 's': 'green'}.get(tile[1:2], 'default'))
                line.append(f'{clean(tile):>3} ', style=style)
            return line

        own = board.players[board.seat] if board.seat in range(4) else None
        hand = list(own.hand) if own else []
        if own and own.drawn in hand:
            hand.remove(own.drawn)
        hand_line = tiles(sorted(hand, key=tile_key))
        if own and own.drawn:
            hand_line.append(' + ', style='dim')
            hand_line.append_text(tiles([own.drawn]))
        if not hand:
            hand_line.append('Waiting for our hand', style='dim')

        wall = '--' if board.wall is None else str(board.wall)
        round_line = text(f'{board.round}  Honba {board.honba}  Sticks {board.sticks}  '
                          f'Wall {wall}  Dora: {" ".join(board.dora) or "--"}')
        if width < 72 or height < 24:
            # Compact terminals retain the essentials without wrapping into
            # scrollback; resizing back restores the full table automatically.
            lines = [text('RIICHI LAB | ' + self.mode, 'bold cyan'), text(status), round_line]
            for i, p in enumerate(board.players):
                lines.append(text(f'{">" if i == board.seat else " "} Seat {i} '
                                  f'{p.score:>7,} {"RIICHI" if p.reach else ""}  {p.name}'))
            lines += [text(f'MORTAL HAND ({board.hand_status})'), hand_line,
                      text(board.result), text(self.notices.warning, 'yellow'),
                      text(summary), text('Ctrl+C quit | Enlarge to 72x24 for rivers', 'dim')]
            return Group(*lines[:height])

        scores = Table(expand=True, box=None, padding=(0, 1), header_style='dim')
        ceiling = score_scale(board.players)
        scores.add_column('Seat', width=9)
        scores.add_column('Player', ratio=1, overflow='ellipsis', no_wrap=True)
        scores.add_column('Tenpai~', width=7, justify='right')
        if width >= 110:
            scores.add_column('Furiten~', width=8, justify='right')
            scores.add_column('Wait tiles~', ratio=2, no_wrap=True, overflow='ellipsis')
        scores.add_column('Score', width=8, justify='right')
        if width >= 110:
            scores.add_column(f'/ {ceiling // 1000}k', width=10)
        if width >= 110:
            scores.add_column('Delta', width=7, justify='right')
        scores.add_column('State / melds', ratio=1, no_wrap=True, overflow='ellipsis')
        for i, p in enumerate(board.players):
            wind = WINDS[(i - board.dealer) % 4] if board.dealer is not None else '-'
            label = f'{i} {wind}' + (' / YOU' if i == board.seat else '')
            state = ('RIICHI ' if p.reach else '') + ('DEALER ' if i == board.dealer else '')
            state += ' '.join(f'{kind}({" ".join(values)})' for kind, values in p.melds)
            estimate = predicted_seats.get(i)
            # The prediction columns occupy the previously empty middle of
            # the scoreboard. Stable percentages, no changing bar colours.
            cells = [text(label, 'cyan' if i == board.seat else ''), text(p.name),
                     text(f'{estimate["tenpai"]:.1%}' if estimate else '--')]
            if width >= 110:
                wait_text = '--'
                if estimate:
                    from riichi_lab.analysis import TILES
                    top = sorted(range(34), key=lambda k: estimate['waits'][k], reverse=True)[
                        :3 if width >= 150 else 2]
                    wait_text = ' '.join(f'{TILES[k]} {estimate["waits"][k]:.1%}' for k in top)
                elif i == board.seat and prediction:
                    from riichi_lab.analysis import TILES
                    available = {tile.rstrip('r') for tile in p.hand} & set(TILES)
                    top = sorted(available, key=lambda t: prediction['any_wait'][TILES.index(t)],
                                 reverse=True)[:2]
                    wait_text = 'Any: ' + ' '.join(
                        f'{t} {prediction["any_wait"][TILES.index(t)]:.1%}' for t in top)
                cells += [text(f'{estimate["furiten"]:.1%}' if estimate else '--'), text(wait_text)]
            cells.append(text(f'{p.score:,}'))
            if width >= 110:
                cells.append(text(score_bar(p.score, ceiling), 'dim cyan'))
            if width >= 110:
                cells.append(text(f'{p.delta:+,}' if p.delta else '--'))
            cells.append(text(state or '--', 'yellow' if p.reach else ''))
            scores.add_row(*cells)

        root = Layout()
        notes_height = 4 if height >= 28 else 3
        root.split_column(Layout(name='header', size=3), Layout(name='round', size=3),
                          Layout(name='scores', size=7), Layout(name='hand', size=3),
                          Layout(name='rivers'), Layout(name='notes', size=notes_height),
                          Layout(name='footer', size=1))
        root['header'].update(panel(text(f'RIICHI LAB / {self.mode} | {status}', 'cyan')))
        root['round'].update(panel(round_line, clean(self.model)))
        root['scores'].update(panel(scores))
        root['hand'].update(panel(hand_line, f'MORTAL HAND ({board.hand_status})'))

        def river(player, count):
            line = Text(no_wrap=True, overflow='ellipsis')
            for discard in player.river[-count:]:
                # Stable notation, no flashing active-seat borders or last-tile backgrounds.
                marker = '*' if discard.reach else ('x' if discard.called else ' ')
                style = 'dim' if discard.drawn or discard.called else ''
                line.append(f'{clean(discard.tile):>3}{marker} ', style=style)
            return line if player.river else text('--', 'dim')

        if width >= 100 and height >= 34:
            root['rivers'].split_column(Layout(name='top'), Layout(name='bottom'))
            for row, seats in (('top', (0, 1)), ('bottom', (2, 3))):
                cards = []
                for i in seats:
                    p = board.players[i]
                    rows = []
                    # Six discards per row, like a physical table. Keep all
                    # discards in state; show the newest rows when space is tight.
                    meld_line = text('Calls: ' + (' | '.join(' '.join(values)
                                     for _, values in p.melds) or '--'), 'dim')
                    capacity = max(1, (height - 17 - notes_height) // 2 - 3)
                    chunks = [p.river[j:j + 6] for j in range(0, len(p.river), 6)]
                    for chunk in chunks[-capacity:]:
                        view = Player(p.name, river=chunk)
                        rows.append(river(view, 6))
                    cards.append(Layout(panel(Group(meld_line, *(rows or [text('--', 'dim')])),
                                              f'SEAT {i} / {len(p.river)} discards')))
                root[row].split_row(*cards)
        else:
            rows = []
            count = max(1, (width - 12) // 5)
            for i, p in enumerate(board.players):
                line = text(f'{i} ({len(p.river):02})  ', 'dim')
                line.append_text(river(p, count))
                rows.append(line)
            root['rivers'].update(Group(*rows))
        notice = self.notices.warning or (self.notices.messages[-1] if self.notices.messages else board.result)
        notes = [text(notice, 'yellow' if self.notices.warning else '')]
        if notes_height > 3:
            notes.append(text(prediction_note, 'dim'))
        root['notes'].update(panel(Group(*notes), board.result))
        root['footer'].update(text(f'{summary} | * riichi x called | Ctrl+C quit', 'dim'))
        return root
