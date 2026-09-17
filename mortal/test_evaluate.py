"""Run with: python -m unittest test_evaluate -v (from mortal/).

Covers the bookkeeping and the statistics; playing games needs libriichi and a
GPU and is checked by reproducing a known evaluation instead.
"""
import json
import os
import tempfile
import unittest

import numpy as np

import evaluate as ev


def games_from(walls):
    """{seed: [a, b, c, d]} back to {(seed, split): rank}."""
    return {(seed, split): rank
            for seed, ranks in walls.items()
            for split, rank in zip(ev.SPLITS, ranks)}


class Walls(unittest.TestCase):
    def test_only_complete_walls_count(self):
        games = games_from({1: [1, 2, 3, 4], 2: [4, 3, 2, 1]})
        del games[(2, 'c')]
        self.assertEqual(ev.walls_of(games), {1: [1, 2, 3, 4]})

    def test_splits_keep_their_seat_order(self):
        games = {(7, 'd'): 4, (7, 'b'): 2, (7, 'a'): 1, (7, 'c'): 3}
        self.assertEqual(ev.walls_of(games), {7: [1, 2, 3, 4]})


class Metrics(unittest.TestCase):
    def test_pt_is_houou_uma(self):
        np.testing.assert_array_equal(ev.metric([1, 2, 3, 4], 'pt'), [90, 45, 0, -135])

    def test_fourth(self):
        np.testing.assert_array_equal(ev.metric([4, 1, 4, 2], 'fourth'), [1, 0, 1, 0])

    def test_level_play_scores_zero(self):
        walls = {s: [1, 2, 3, 4] for s in range(10)}
        self.assertAlmostEqual(ev.summarize(walls, range(10), 'pt')['mean'], 0.0)
        self.assertAlmostEqual(ev.summarize(walls, range(10), 'rank')['mean'], 2.5)


class Paired(unittest.TestCase):
    def test_difference_and_error_are_over_walls(self):
        a = {0: [1, 1, 1, 1], 1: [2, 2, 2, 2], 2: [1, 2, 1, 2]}
        b = {0: [2, 2, 2, 2], 1: [2, 2, 2, 2], 2: [2, 2, 2, 2]}
        res = ev.paired(a, b, [0, 1, 2], 'rank', reps=0)
        d = np.array([-1.0, 0.0, -0.5])
        self.assertAlmostEqual(res['diff'], d.mean())
        self.assertAlmostEqual(res['se'], d.std(ddof=1) / np.sqrt(3))
        self.assertEqual(res['walls'], 3)

    def test_identical_games_have_no_error(self):
        a = {s: [1, 2, 3, 4] for s in range(5)}
        res = ev.paired(a, a, range(5), 'pt', reps=100)
        self.assertEqual(res['diff'], 0.0)
        self.assertEqual(ev.in_se(res), 'n/a')

    def test_bootstrap_brackets_the_mean_and_repeats(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0.3, 1.0, size=400)
        lo, hi = ev.bootstrap_ci(x, reps=2000)
        self.assertLess(lo, x.mean())
        self.assertGreater(hi, x.mean())
        # Close to mean +- 1.96 se for a sample this size.
        se = x.std(ddof=1) / np.sqrt(len(x))
        self.assertAlmostEqual(hi - lo, 2 * 1.96 * se, delta=0.2 * se * 2 * 1.96)
        self.assertEqual((lo, hi), ev.bootstrap_ci(x, reps=2000))

    def test_games_to_detect(self):
        # sd 1 per wall, effect 0.1: (2.8 / 0.1)^2 = 784.9 walls -> 785 walls.
        self.assertEqual(ev.games_to_detect(1.0, 0.1), 4 * 785)
        self.assertTrue(np.isnan(ev.games_to_detect(float('nan'), 0.1)))


class Chunks(unittest.TestCase):
    def test_chunks_tile_the_set(self):
        s = ev.WALL_SETS['legacy']
        cs = ev.chunks(s, 300)
        self.assertEqual(cs, [(10000, 300), (10300, 300), (10600, 300), (10900, 100)])
        self.assertEqual(ev.chunks(s, 250, limit=260), [(10000, 250), (10250, 10)])

    def test_sets_do_not_overlap(self):
        keys = [s.key for s in ev.WALL_SETS.values()]
        self.assertEqual(len(keys), len(set(keys)))

    def test_legacy_is_what_test_play_dealt(self):
        # player.TestPlayer: seeds from 10000, key 0x2000, games / 4 of them.
        s = ev.WALL_SETS['legacy']
        self.assertEqual((s.first_seed, s.key, s.games), (10000, 0x2000, 4000))


class Storage(unittest.TestCase):
    def test_read_games_takes_finished_chunks_only(self):
        with tempfile.TemporaryDirectory() as d:
            done = os.path.join(d, ev.chunk_name(0, 1))
            os.makedirs(done)
            with open(os.path.join(done, ev.SUMMARY), 'w') as f:
                json.dump({'ranks': {'0_a': 1, '0_b': 2, '0_c': 3, '0_d': 4}}, f)
            # An interrupted chunk has logs but no summary until it is renamed in.
            os.makedirs(os.path.join(d, ev.chunk_name(1, 1) + '.partial'))
            self.assertEqual(ev.walls_of(ev.read_games(d)), {0: [1, 2, 3, 4]})

    def test_a_wider_chunk_replaces_the_ones_inside_it(self):
        def finish(d, first, count):
            chunk = os.path.join(d, ev.chunk_name(first, count))
            os.makedirs(chunk)
            ranks = {f'{s}_{x}': 1 + i for s in range(first, first + count)
                     for i, x in enumerate(ev.SPLITS)}
            with open(os.path.join(chunk, ev.SUMMARY), 'w') as f:
                json.dump({'ranks': ranks}, f)

        with tempfile.TemporaryDirectory() as d:
            finish(d, 0, 1)
            finish(d, 1, 1)
            finish(d, 3, 2)
            finish(d, 0, 4)
            self.assertEqual(ev.logged_games(d), 4 * (1 + 1 + 2 + 4))
            ev.drop_covered(d, 0, 4)
            # [3, 5) sticks out past the new chunk, so it stays, overlap and all.
            self.assertEqual(ev.finished_chunks(d), [ev.chunk_name(0, 4), ev.chunk_name(3, 2)])
            self.assertEqual(ev.logged_games(d) - len(ev.read_games(d)), 4)

    def test_summary_files_are_not_game_logs(self):
        # Stat.from_dir reads every *.json and *.json.gz as a game.
        for name in (ev.SUMMARY, ev.META):
            self.assertFalse(name.endswith('.json') or name.endswith('.json.gz'), name)

    def test_spec(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, 'mortal.pth')
            with open(f, 'wb') as out:
                out.write(b'weights')
            plain = ev.Spec.parse(f)
            self.assertEqual((plain.label, plain.part), ('mortal', ''))
            ema = ev.Spec.parse(f'late={f}#ema')
            self.assertEqual((ema.label, ema.part), ('late', 'ema'))
            self.assertEqual(ema.ident, plain.ident + '-ema')
            self.assertEqual(len(plain.ident), 16)
            with self.assertRaises(SystemExit):
                ev.Spec.parse(f + '#best')


if __name__ == '__main__':
    unittest.main()
