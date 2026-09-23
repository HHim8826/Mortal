use super::{CALC_ALL, CALC_SHANTEN_FN};
use super::tile::{DiscardTile, DrawTile, RequiredTile};
use crate::algo::shanten::Neighbours;
use crate::tile::Tile;
use crate::{must_tile, t, tu8};

use std::hash::{Hash, Hasher};
use tinyvec::ArrayVec;

/// Mutable state of both the hand and the board.
///
/// Changed only through its methods, so that `key` always describes the
/// fields beside it.
#[derive(Clone, PartialEq, Eq)]
pub(super) struct State {
    // hand
    pub(super) tehai: [u8; 34],
    pub(super) akas_in_hand: [bool; 3],

    // global
    pub(super) tiles_in_wall: [u8; 34],
    pub(super) akas_in_wall: [bool; 3],
    pub(super) n_extra_tsumo: u8,

    /// A Zobrist hash of everything above, kept up to date by the methods that
    /// change it. The search memoises on `State`, and the derived hash went
    /// over all 75 bytes of it, twice for every new node (the lookup that
    /// missed, then the insert) -- about a tenth of an observation's time.
    key: u64,
}

impl Hash for State {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.key);
    }
}

/// Deterministic pseudo-random numbers for the key, one per (tile, count) of
/// the hand and of the wall, per red five in either, and per extra tsumo.
const fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

const fn table(salt: u64) -> [[u64; 5]; 34] {
    let mut out = [[0; 5]; 34];
    let mut i = 0;
    while i < 34 {
        let mut c = 0;
        while c < 5 {
            out[i][c] = splitmix(salt ^ ((i as u64) << 8) ^ c as u64);
            c += 1;
        }
        i += 1;
    }
    out
}

const HAND: [[u64; 5]; 34] = table(0x68_61_6e_64 << 16);
const WALL: [[u64; 5]; 34] = table(0x77_61_6c_6c << 16);
const AKA_HAND: [u64; 3] = [splitmix(0xa1), splitmix(0xa2), splitmix(0xa3)];
const AKA_WALL: [u64; 3] = [splitmix(0xb1), splitmix(0xb2), splitmix(0xb3)];

const fn extra(n: u8) -> u64 {
    splitmix(0xe0_0000 ^ n as u64)
}

/// Mutable state of both the hand and the board.
#[derive(Clone)]
pub struct InitState {
    // hand
    pub tehai: [u8; 34],
    pub akas_in_hand: [bool; 3],

    // global
    pub tiles_seen: [u8; 34],
    pub akas_seen: [bool; 3],
}

impl From<InitState> for State {
    fn from(
        InitState {
            tehai,
            akas_in_hand,
            tiles_seen,
            akas_seen,
        }: InitState,
    ) -> Self {
        let mut tiles_in_wall = tiles_seen;
        let mut akas_in_wall = akas_seen;
        tiles_in_wall.iter_mut().for_each(|v| *v = 4 - *v);
        akas_in_wall.iter_mut().for_each(|v| *v = !*v);
        let mut key = extra(0);
        for tid in 0..34 {
            key ^= HAND[tid][tehai[tid] as usize] ^ WALL[tid][tiles_in_wall[tid] as usize];
        }
        for i in 0..3 {
            if akas_in_hand[i] {
                key ^= AKA_HAND[i];
            }
            if akas_in_wall[i] {
                key ^= AKA_WALL[i];
            }
        }
        Self {
            tehai,
            akas_in_hand,
            tiles_in_wall,
            akas_in_wall,
            n_extra_tsumo: 0,
            key,
        }
    }
}

impl State {
    const fn set_hand(&mut self, tid: usize, count: u8) {
        self.key ^= HAND[tid][self.tehai[tid] as usize] ^ HAND[tid][count as usize];
        self.tehai[tid] = count;
    }

    const fn set_wall(&mut self, tid: usize, count: u8) {
        self.key ^= WALL[tid][self.tiles_in_wall[tid] as usize] ^ WALL[tid][count as usize];
        self.tiles_in_wall[tid] = count;
    }

    const fn set_aka_in_hand(&mut self, i: usize, has: bool) {
        if self.akas_in_hand[i] != has {
            self.key ^= AKA_HAND[i];
            self.akas_in_hand[i] = has;
        }
    }

    const fn set_aka_in_wall(&mut self, i: usize, has: bool) {
        if self.akas_in_wall[i] != has {
            self.key ^= AKA_WALL[i];
            self.akas_in_wall[i] = has;
        }
    }

    pub(super) const fn add_extra_tsumo(&mut self) {
        self.key ^= extra(self.n_extra_tsumo) ^ extra(self.n_extra_tsumo + 1);
        self.n_extra_tsumo += 1;
    }

    pub(super) const fn remove_extra_tsumo(&mut self) {
        self.key ^= extra(self.n_extra_tsumo) ^ extra(self.n_extra_tsumo - 1);
        self.n_extra_tsumo -= 1;
    }

    pub(super) const fn discard(&mut self, tile: Tile) {
        let tid = tile.deaka().as_usize();
        self.set_hand(tid, self.tehai[tid] - 1);
        match tile.as_u8() {
            tu8!(5mr) => self.set_aka_in_hand(0, false),
            tu8!(5pr) => self.set_aka_in_hand(1, false),
            tu8!(5sr) => self.set_aka_in_hand(2, false),
            _ => (),
        }
    }

    pub(super) const fn undo_discard(&mut self, tile: Tile) {
        let tid = tile.deaka().as_usize();
        self.set_hand(tid, self.tehai[tid] + 1);
        match tile.as_u8() {
            tu8!(5mr) => self.set_aka_in_hand(0, true),
            tu8!(5pr) => self.set_aka_in_hand(1, true),
            tu8!(5sr) => self.set_aka_in_hand(2, true),
            _ => (),
        }
    }

    pub(super) const fn deal(&mut self, tile: Tile) {
        let tid = tile.deaka().as_usize();
        self.set_wall(tid, self.tiles_in_wall[tid] - 1);
        match tile.as_u8() {
            tu8!(5mr) => self.set_aka_in_wall(0, false),
            tu8!(5pr) => self.set_aka_in_wall(1, false),
            tu8!(5sr) => self.set_aka_in_wall(2, false),
            _ => (),
        }
        self.undo_discard(tile);
    }

    pub(super) const fn undo_deal(&mut self, tile: Tile) {
        self.discard(tile);
        let tid = tile.deaka().as_usize();
        self.set_wall(tid, self.tiles_in_wall[tid] + 1);
        match tile.as_u8() {
            tu8!(5mr) => self.set_aka_in_wall(0, true),
            tu8!(5pr) => self.set_aka_in_wall(1, true),
            tu8!(5sr) => self.set_aka_in_wall(2, true),
            _ => (),
        }
    }

    pub(super) fn get_discard_tiles(
        &self,
        shanten: i8,
        tehai_len_div3: u8,
    ) -> ArrayVec<[DiscardTile; 14]> {
        let mut discard_tiles = ArrayVec::default();

        let tehai = self.tehai;
        let neighbours = Neighbours::new(&tehai, tehai_len_div3, CALC_ALL);
        for tid in 0..34 {
            if tehai[tid] == 0 {
                continue;
            }

            let shanten_after = neighbours.without(tid);

            let shanten_diff = shanten_after - shanten;

            let tile = match tid as u8 {
                tu8!(5m) if self.akas_in_hand[0] && tehai[tid] == 1 => t!(5mr),
                tu8!(5p) if self.akas_in_hand[1] && tehai[tid] == 1 => t!(5pr),
                tu8!(5s) if self.akas_in_hand[2] && tehai[tid] == 1 => t!(5sr),
                _ => must_tile!(tid),
            };

            discard_tiles.push(DiscardTile { tile, shanten_diff });
        }

        discard_tiles
    }

    pub(super) fn get_draw_tiles(
        &self,
        shanten: i8,
        tehai_len_div3: u8,
    ) -> ArrayVec<[DrawTile; 37]> {
        let mut draw_tiles = ArrayVec::default();

        let neighbours = Neighbours::new(&self.tehai, tehai_len_div3, CALC_ALL);
        for (tid, &count) in self.tiles_in_wall.iter().enumerate() {
            if count == 0 {
                continue;
            }

            let shanten_after = neighbours.with(tid);

            let shanten_diff = shanten_after - shanten;

            let tile = must_tile!(tid);
            match (tid as u8, self.akas_in_wall) {
                (tu8!(5m), [true, _, _]) | (tu8!(5p), [_, true, _]) | (tu8!(5s), [_, _, true]) => {
                    if count >= 2 {
                        draw_tiles.push(DrawTile {
                            tile,
                            count: count - 1,
                            shanten_diff,
                        });
                    }
                    draw_tiles.push(DrawTile {
                        tile: tile.akaize(),
                        count: 1,
                        shanten_diff,
                    });
                }
                _ => draw_tiles.push(DrawTile {
                    tile,
                    count,
                    shanten_diff,
                }),
            }
        }

        draw_tiles
    }

    pub(super) fn get_required_tiles(&self, tehai_len_div3: u8) -> ArrayVec<[RequiredTile; 34]> {
        let shanten = CALC_SHANTEN_FN(&self.tehai, tehai_len_div3);
        let neighbours = Neighbours::new(&self.tehai, tehai_len_div3, CALC_ALL);
        let mut required_tiles = ArrayVec::default();

        for (tid, &count) in self.tiles_in_wall.iter().enumerate() {
            if count == 0 {
                continue;
            }

            let shanten_after = neighbours.with(tid);

            if shanten_after < shanten {
                required_tiles.push(RequiredTile {
                    tile: must_tile!(tid),
                    count,
                });
            }
        }

        required_tiles
    }

    pub(super) fn sum_left_tiles(&self) -> u8 {
        self.tiles_in_wall.iter().sum()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    fn recomputed(s: &State) -> u64 {
        let mut key = extra(s.n_extra_tsumo);
        for tid in 0..34 {
            key ^= HAND[tid][s.tehai[tid] as usize] ^ WALL[tid][s.tiles_in_wall[tid] as usize];
        }
        for i in 0..3 {
            if s.akas_in_hand[i] {
                key ^= AKA_HAND[i];
            }
            if s.akas_in_wall[i] {
                key ^= AKA_WALL[i];
            }
        }
        key
    }

    #[test]
    fn the_key_follows_every_change() {
        let mut rng = ChaCha8Rng::seed_from_u64(0x2b);
        for _ in 0..2_000 {
            let mut tehai = [0; 34];
            let mut seen = [0; 34];
            for _ in 0..13 {
                let t = rng.random_range(0..34);
                if tehai[t] < 4 {
                    tehai[t] += 1;
                    seen[t] += 1;
                }
            }
            let mut state = State::from(InitState {
                tehai,
                akas_in_hand: [false; 3],
                tiles_seen: seen,
                akas_seen: [false; 3],
            });
            assert_eq!(state.key, recomputed(&state));
            let start = state.clone();

            // A path down the search and back up it, as the search walks.
            let mut undo = vec![];
            for _ in 0..rng.random_range(1..12) {
                match rng.random_range(0..3) {
                    0 => {
                        let t = rng.random_range(0..37);
                        let tile = must_tile!(t);
                        let tid = tile.deaka().as_usize();
                        let red_ok = t < 34 || state.akas_in_wall[t - 34];
                        if state.tiles_in_wall[tid] > 0 && state.tehai[tid] < 4 && red_ok {
                            state.deal(tile);
                            undo.push((0, tile));
                        }
                    }
                    1 => {
                        let tid = rng.random_range(0..34);
                        if state.tehai[tid] > 0 {
                            let tile = must_tile!(tid);
                            state.discard(tile);
                            undo.push((1, tile));
                        }
                    }
                    _ => {
                        state.add_extra_tsumo();
                        undo.push((2, t!(?)));
                    }
                }
                assert_eq!(state.key, recomputed(&state));
            }
            while let Some((op, tile)) = undo.pop() {
                match op {
                    0 => state.undo_deal(tile),
                    1 => state.undo_discard(tile),
                    _ => state.remove_extra_tsumo(),
                }
                assert_eq!(state.key, recomputed(&state));
            }
            assert!(state == start);
        }
    }
}
