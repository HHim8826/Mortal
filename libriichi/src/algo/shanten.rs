//! Rust port of tomohxx's C++ implementation of Shanten Number Calculator.
//!
//! Source: <https://github.com/tomohxx/shanten-number-calculator/>

use crate::tuz;
use std::io::prelude::*;
use std::sync::LazyLock;

use flate2::read::GzDecoder;

const JIHAI_TABLE_SIZE: usize = 78_032;
const SUHAI_TABLE_SIZE: usize = 1_940_777;

static JIHAI_TABLE: LazyLock<Vec<[u8; 10]>> = LazyLock::new(|| {
    read_table(
        include_bytes!("data/shanten_jihai.bin.gz"),
        JIHAI_TABLE_SIZE,
    )
});
static SUHAI_TABLE: LazyLock<Vec<[u8; 10]>> = LazyLock::new(|| {
    read_table(
        include_bytes!("data/shanten_suhai.bin.gz"),
        SUHAI_TABLE_SIZE,
    )
});

fn read_table(gzipped: &[u8], length: usize) -> Vec<[u8; 10]> {
    let mut gz = GzDecoder::new(gzipped);
    let mut raw = vec![];
    gz.read_to_end(&mut raw).unwrap();

    let mut ret = Vec::with_capacity(length);
    let mut entry = [0; 10];
    for (i, b) in raw.into_iter().enumerate() {
        entry[i * 2 % 10] = b & 0b1111;
        entry[i * 2 % 10 + 1] = (b >> 4) & 0b1111;
        if (i + 1) % 5 == 0 {
            ret.push(entry);
        }
    }
    assert_eq!(ret.len(), length);

    ret
}

pub fn ensure_init() {
    assert_eq!(JIHAI_TABLE.len(), JIHAI_TABLE_SIZE);
    assert_eq!(SUHAI_TABLE.len(), SUHAI_TABLE_SIZE);
}

fn add_suhai(lhs: &mut [u8; 10], index: usize, m: usize) {
    let tab = SUHAI_TABLE.get(index).copied().unwrap_or_default();

    for j in (5..=(5 + m)).rev() {
        let mut sht = (lhs[j] + tab[0]).min(lhs[0] + tab[j]);
        for k in 5..j {
            sht = sht.min(lhs[k] + tab[j - k]).min(lhs[j - k] + tab[k]);
        }
        lhs[j] = sht;
    }

    for j in (0..=m).rev() {
        let mut sht = lhs[j] + tab[0];
        for k in 0..j {
            sht = sht.min(lhs[k] + tab[j - k]);
        }
        lhs[j] = sht;
    }
}

fn add_jihai(lhs: &mut [u8; 10], index: usize, m: usize) {
    let tab = JIHAI_TABLE.get(index).copied().unwrap_or_default();

    let j = m + 5;
    let mut sht = (lhs[j] + tab[0]).min(lhs[0] + tab[j]);
    for k in 5..j {
        sht = sht.min(lhs[k] + tab[j - k]).min(lhs[j - k] + tab[k]);
    }
    lhs[j] = sht;
}

fn sum_tiles(tiles: &[u8]) -> usize {
    tiles.iter().fold(0, |acc, &x| acc * 5 + x as usize)
}

/// `len_div3` must be within [0, 4].
#[must_use]
pub fn calc_normal(tiles: &[u8; 34], len_div3: u8) -> i8 {
    let len_div3 = len_div3 as usize;

    let mut ret = SUHAI_TABLE
        .get(sum_tiles(&tiles[..9]))
        .copied()
        .unwrap_or_default();
    add_suhai(&mut ret, sum_tiles(&tiles[9..2 * 9]), len_div3);
    add_suhai(&mut ret, sum_tiles(&tiles[2 * 9..3 * 9]), len_div3);
    add_jihai(&mut ret, sum_tiles(&tiles[3 * 9..]), len_div3);

    (ret[5 + len_div3] as i8) - 1
}

#[must_use]
pub fn calc_chitoi(tiles: &[u8; 34]) -> i8 {
    let mut pairs = 0;
    let mut kinds = 0;
    tiles.iter().filter(|&&c| c > 0).for_each(|&c| {
        kinds += 1;
        if c >= 2 {
            pairs += 1;
        }
    });

    let redunct = 7_u8.saturating_sub(kinds) as i8;
    7 - pairs + redunct - 1
}

#[must_use]
pub fn calc_kokushi(tiles: &[u8; 34]) -> i8 {
    let mut pairs = 0;
    let mut kinds = 0;

    tuz![1m, 9m, 1p, 9p, 1s, 9s, E, S, W, N, P, F, C]
        .iter()
        .map(|&i| tiles[i])
        .filter(|&c| c > 0)
        .for_each(|c| {
            kinds += 1;
            if c >= 2 {
                pairs += 1;
            }
        });

    let redunct = (pairs > 0) as i8;
    14 - kinds - redunct - 1
}

/// The shanten of every hand one tile away from a given one.
///
/// The single-player EV search asks this at every node -- what drawing each
/// of 34 tiles, or discarding each of 14, does to the shanten -- and it was
/// most of the cost of a v4 observation: 57% of the time in the shanten
/// tables, all of it recombining the three suits the tile did not touch.
///
/// Combining suits is a min-plus product of their tables (with the pair as
/// a variable that squares to nothing), which is associative and
/// commutative, and truncating it at `len_div3` melds commutes with it. So
/// the other three suits are combined once, here, and each neighbour costs
/// one table lookup and the one entry of one product the answer is read
/// from. The result is exactly `calc_all` (or `calc_normal`) of the changed
/// hand, not an approximation of it.
pub struct Neighbours {
    tiles: [u8; 34],
    len_div3: u8,
    all: bool,
    index: [usize; 4],
    rest: [[u8; 10]; 4],
    // What chitoi and kokushi count, so a neighbour adjusts one tile of them
    // instead of recounting the hand.
    kinds: i8,
    pairs: i8,
    yaochu_kinds: i8,
    yaochu_pairs: i8,
}

/// The thirteen tiles kokushi counts: terminals and honours.
const fn is_yaochu(tid: usize) -> bool {
    matches!(tid, 0 | 8 | 9 | 17 | 18 | 26) || tid >= 27
}

/// 5^(8 - i): the weight of position i in `sum_tiles`' index of nine tiles.
/// The seven honours are positions 2..9 of the same sequence.
const POW5: [usize; 9] = [390_625, 78_125, 15_625, 3_125, 625, 125, 25, 5, 1];

fn suhai(index: usize) -> [u8; 10] {
    SUHAI_TABLE.get(index).copied().unwrap_or_default()
}

fn jihai(index: usize) -> [u8; 10] {
    JIHAI_TABLE.get(index).copied().unwrap_or_default()
}

/// The product `add_suhai` computes in place, for any two tables.
fn product(lhs: &[u8; 10], rhs: &[u8; 10], m: usize) -> [u8; 10] {
    let mut out = *lhs;
    for j in (5..=(5 + m)).rev() {
        let mut sht = (lhs[j] + rhs[0]).min(lhs[0] + rhs[j]);
        for k in 5..j {
            sht = sht.min(lhs[k] + rhs[j - k]).min(lhs[j - k] + rhs[k]);
        }
        out[j] = sht;
    }
    for j in (0..=m).rev() {
        let mut sht = lhs[j] + rhs[0];
        for k in 0..j {
            sht = sht.min(lhs[k] + rhs[j - k]);
        }
        out[j] = sht;
    }
    out
}

/// The one entry of the product `calc_normal` reads its answer from.
fn final_entry(lhs: &[u8; 10], rhs: &[u8; 10], m: usize) -> u8 {
    let j = m + 5;
    let mut sht = (lhs[j] + rhs[0]).min(lhs[0] + rhs[j]);
    for k in 5..j {
        sht = sht.min(lhs[k] + rhs[j - k]).min(lhs[j - k] + rhs[k]);
    }
    sht
}

impl Neighbours {
    /// `all` answers as `calc_all` does, otherwise as `calc_normal`.
    #[must_use]
    pub fn new(tiles: &[u8; 34], len_div3: u8, all: bool) -> Self {
        let m = len_div3 as usize;
        let index = [
            sum_tiles(&tiles[..9]),
            sum_tiles(&tiles[9..2 * 9]),
            sum_tiles(&tiles[2 * 9..3 * 9]),
            sum_tiles(&tiles[3 * 9..]),
        ];
        let [man, pin, sou, ji] = [suhai(index[0]), suhai(index[1]), suhai(index[2]), jihai(index[3])];
        // Six products rather than eight: each suit's "rest" is one of two
        // pairs times one suit of the other.
        let man_pin = product(&man, &pin, m);
        let sou_ji = product(&sou, &ji, m);
        let rest = [
            product(&pin, &sou_ji, m),
            product(&man, &sou_ji, m),
            product(&man_pin, &ji, m),
            product(&man_pin, &sou, m),
        ];

        let (mut kinds, mut pairs, mut yaochu_kinds, mut yaochu_pairs) = (0, 0, 0, 0);
        for (tid, &c) in tiles.iter().enumerate() {
            let (kind, pair) = ((c > 0) as i8, (c >= 2) as i8);
            kinds += kind;
            pairs += pair;
            if is_yaochu(tid) {
                yaochu_kinds += kind;
                yaochu_pairs += pair;
            }
        }
        Self {
            tiles: *tiles,
            len_div3,
            all,
            index,
            rest,
            kinds,
            pairs,
            yaochu_kinds,
            yaochu_pairs,
        }
    }

    /// The shanten of the hand with one more of tile `tid`.
    #[must_use]
    pub fn with(&self, tid: usize) -> i8 {
        self.changed(tid, true)
    }

    /// The shanten of the hand with one fewer of tile `tid`.
    #[must_use]
    pub fn without(&self, tid: usize) -> i8 {
        self.changed(tid, false)
    }

    fn changed(&self, tid: usize, add: bool) -> i8 {
        let m = self.len_div3 as usize;
        let suit = tid / 9;
        let step = if suit < 3 { POW5[tid % 9] } else { POW5[tid - 27 + 2] };
        let index = if add {
            self.index[suit] + step
        } else {
            self.index[suit] - step
        };
        let table = if suit < 3 { suhai(index) } else { jihai(index) };
        let normal = final_entry(&self.rest[suit], &table, m) as i8 - 1;
        if !self.all || normal <= 0 || self.len_div3 < 4 {
            return normal;
        }

        // The same counts calc_chitoi and calc_kokushi make, moved by one tile.
        let c = self.tiles[tid];
        let (d_kind, d_pair) = if add {
            ((c == 0) as i8, (c == 1) as i8)
        } else {
            (-((c == 1) as i8), -((c == 2) as i8))
        };
        let (kinds, pairs) = (self.kinds + d_kind, self.pairs + d_pair);
        let chitoi = 7 - pairs + (7 - kinds).max(0) - 1;
        let shanten = normal.min(chitoi);
        if shanten > 0 {
            let (mut yaochu_kinds, mut yaochu_pairs) = (self.yaochu_kinds, self.yaochu_pairs);
            if is_yaochu(tid) {
                yaochu_kinds += d_kind;
                yaochu_pairs += d_pair;
            }
            let kokushi = 14 - yaochu_kinds - (yaochu_pairs > 0) as i8 - 1;
            shanten.min(kokushi)
        } else {
            shanten
        }
    }
}

#[must_use]
pub fn calc_all(tiles: &[u8; 34], len_div3: u8) -> i8 {
    let mut shanten = calc_normal(tiles, len_div3);
    if shanten <= 0 || len_div3 < 4 {
        return shanten;
    }

    shanten = shanten.min(calc_chitoi(tiles));
    if shanten > 0 {
        shanten.min(calc_kokushi(tiles))
    } else {
        shanten
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::hand::hand;

    #[test]
    fn calc_3n_plus_1() {
        let tehai = hand("1111m 333p 222s 444z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 1);
        let tehai = hand("147m 258p 369s 1234z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 6);
        let tehai = hand("468m 33346p 7s").unwrap();
        assert_eq!(calc_all(&tehai, 3), 2);
        let tehai = hand("147m 258p 3s").unwrap();
        assert_eq!(calc_all(&tehai, 2), 4);
        let tehai = hand("4455s").unwrap();
        assert_eq!(calc_all(&tehai, 1), 0);
        let tehai = hand("7z").unwrap();
        assert_eq!(calc_all(&tehai, 0), 0);
        let tehai = hand("15559m 19p 19s 1234z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 3);
        let tehai = hand("9999m 6677p 88s 355z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 2);
        let tehai = hand("19m 19p 159s 123456z").unwrap();
        assert_eq!(calc_all(&tehai, 4), 1);
    }

    #[test]
    fn neighbours_are_exactly_the_full_calculation() {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(0x5ea7);
        let mut checked = 0;
        for _ in 0..20_000 {
            // Any length a hand can have, 3n+1 and 3n+2, drawn from a real
            // wall of four of each tile.
            let len_div3 = rng.random_range(0..=4_u8);
            let len = len_div3 as usize * 3 + rng.random_range(1..=2);
            let mut wall: Vec<usize> = (0..34).flat_map(|t| [t; 4]).collect();
            wall.shuffle(&mut rng);
            let mut tiles = [0_u8; 34];
            for &t in &wall[..len] {
                tiles[t] += 1;
            }

            for all in [true, false] {
                let full = if all { calc_all } else { calc_normal };
                let nb = Neighbours::new(&tiles, len_div3, all);
                for tid in 0..34 {
                    if tiles[tid] < 4 && len % 3 == 1 {
                        let mut t = tiles;
                        t[tid] += 1;
                        assert_eq!(nb.with(tid), full(&t, len_div3), "{tiles:?} + {tid}");
                        checked += 1;
                    }
                    if tiles[tid] > 0 {
                        let mut t = tiles;
                        t[tid] -= 1;
                        assert_eq!(nb.without(tid), full(&t, len_div3), "{tiles:?} - {tid}");
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 500_000, "{checked}");
    }

    #[test]
    fn calc_3n_plus_2() {
        let tehai = hand("2344456m 14p 127s 2z 7p").unwrap();
        assert_eq!(calc_all(&tehai, 4), 3);
        let tehai = hand("2344456m 14p 127s 2z 5p").unwrap();
        assert_eq!(calc_all(&tehai, 4), 2);
        let tehai = hand("344455667p 1139s 9m").unwrap();
        assert_eq!(calc_all(&tehai, 4), 2);
        let tehai = hand("344455667p 1139s 9p").unwrap();
        assert_eq!(calc_all(&tehai, 4), 1);
        let tehai = hand("122334m 678p 37s 22z 5s").unwrap();
        assert_eq!(calc_all(&tehai, 4), 0);
        let tehai = hand("122334m 678p 12s 22z 4s").unwrap();
        assert_eq!(calc_all(&tehai, 4), 0);
        let tehai = hand("12223456m 78889p 2m").unwrap();
        assert_eq!(calc_all(&tehai, 4), -1);
        let tehai = hand("34778p").unwrap();
        assert_eq!(calc_all(&tehai, 1), 0);
        let tehai = hand("34s").unwrap();
        assert_eq!(calc_all(&tehai, 0), 0);
        let tehai = hand("55m").unwrap();
        assert_eq!(calc_all(&tehai, 0), -1);
    }
}
