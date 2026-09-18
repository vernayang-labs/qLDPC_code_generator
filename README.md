# Hypergraph product qLDPC codes

Construct CSS [hypergraph product (HGP)](https://errorcorrectionzoo.org/c/hypergraph_product) codes from classical seed codes, using the [`qldpc`](https://github.com/qLDPCOrg/qLDPC) library.

An HGP code is built from two classical linear codes \(A\) and \(B\) with parity-check matrices \(H_1\) and \(H_2\):

\[
H_X = \bigl[ H_1 \otimes I_{n_2} \ \big|\  I_{m_1} \otimes H_2^\top \bigr],
\qquad
H_Z = \bigl[ I_{n_1} \otimes H_2 \ \big|\  H_1^\top \otimes I_{m_2} \bigr],
\]

where \((m_i, n_i) = H_i.\mathrm{shape}\). If \(B\) is omitted, the construction uses \(A\) twice (the homological square). This is the check-matrix product of Tillich and Zémor [[3]](#references).

Built-in MacKay–Neal and semitopological seeds reproduce the constant-rate and augmented families in Roffe *et al.* (2020) [[1]](#references). The length-16 MacKay matrix is the published seed of Example 3.2 in Roffe *et al.* (2023) [[2]](#references).

## Setup

Python 3.10+ is required (this is also the minimum for `qldpc`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install qldpc numpy sympy
```

## Usage

```bash
python construct_hgp_code.py
python construct_hgp_code.py --code-a hamming:3
python construct_hgp_code.py --code-a repetition:5 --code-b repetition:4
python construct_hgp_code.py --code-a 'cyclic:7,1+x+x**3' --code-b hamming:3
python construct_hgp_code.py --code-a mackay:16 --save-dir out/regular/hgp_400_16_6
python construct_hgp_code.py --code-a semitopo:0 --save-dir out/semitopo/hgp_13_5_2
python construct_hgp_code.py --code-a random:10,6 --seed 0 --no-distance
```

`--save-dir` writes `hx.txt`, `hz.txt`, `logical_x.txt`, `logical_z.txt`, NumPy copies of \(H_X\) and \(H_Z\), and `metadata.json`.

Use `--no-distance` to skip exact distance (useful for large or random seeds). `--bound N` replaces exact \(d\) with an upper bound from `N` QDistRnd trials.

### Seed specs

| Spec | Classical seed |
| --- | --- |
| `repetition:N` | Length-\(N\) repetition code |
| `ring:N` | Length-\(N\) ring (cycle) code |
| `hamming:M` | Hamming code of rank \(M\) (binary length \(2^M-1\)) |
| `extended-hamming:M` | Extended Hamming code |
| `simplex:K` | Simplex code |
| `reed-muller:R,M` | Reed–Muller code |
| `cyclic:N,POLY` | Cyclic code, e.g. `cyclic:7,1+x+x**3` |
| `mackay:N` | (3,4)-regular MacKay–Neal seed; \(N\in\{16,20,24\}\) [[1](#references),[2](#references),[4](#references)] |
| `semitopo:G` | (2,3)-LDPC parent with edge-augmentation \(g\) [[1]](#references) |
| `random:N,M` | Random code with \(N\) bits and \(M\) checks |
| `matrix:PATH` | Integer parity-check matrix from a file |

Default: `--code-a hamming:3` (the \([7,4,3]\) Hamming code), with \(B = A\).

## Constructed families

### Constant-rate MacKay–Neal HGP codes

(3,4)-regular classical seeds with girth \(>4\) and full-rank \(H\), so \(H^\top\) encodes nothing (\(d^T=\infty\)). The hypergraph product is the rate-\(0.04\) (8,7)-QLDPC family of Table I in [[1]](#references). The \(n=16\) matrix is Example 3.2 / Appendix A of [[2]](#references); \(n=20\) and \(n=24\) are ensemble-equivalent (3,4)-regular seeds with the same classical \([n,k,d]\).

| Seed | Classical \([n,k,d]\) | HGP \([[n,k,d]]\) | Output |
| --- | --- | --- | --- |
| `mackay:16` | \([16,4,6]\) | \([[400,16,6]]\) | `out/regular/hgp_400_16_6/` |
| `mackay:20` | \([20,5,8]\) | \([[625,25,8]]\) | `out/regular/hgp_625_25_8/` |
| `mackay:24` | \([24,6,10]\) | \([[900,36,10]]\) | `out/regular/hgp_900_36_10/` |

### Semitopological HGP codes

The parent is the \([3,2,2]\) (2,3)-LDPC code with \(H=\bigl[\begin{smallmatrix}1&1&1\\1&1&1\end{smallmatrix}\bigr]\). Edge-augmenting every Tanner edge by a length-\(g\) repetition chain, then taking the hypergraph product, yields the family \(\mathrm{HGP}(C_H^{*g})\) of Table II / Fig. 2 in [[1]](#references).

| Seed | Paper \([[n,k,d]]\) [[1]](#references) | This repo | Output |
| --- | --- | --- | --- |
| `semitopo:0` | \([[13,5,2]]\) | \([[13,5,2]]\) | `out/semitopo/hgp_13_5_2/` |
| `semitopo:1` | \([[145,5,6]]\) | \([[145,5,6]]\) | `out/semitopo/hgp_145_5_6/` |
| `semitopo:2` | \([[421,5,10]]\) | \([[421,13,3]]\) | `out/semitopo/hgp_421_5_10/` |

For \(g=2\), the block length matches Table II, but the logical dimension and distance differ from the published parameters (the paper’s classical seed is \([15,2,10]\); this implementation currently produces a \([15,3,3]\) seed). Directory names follow the table in [[1]](#references); `metadata.json` records the values computed here.

## References

1. Joschka Roffe, David R. White, Simon Burton, and Earl Campbell, [Decoding across the quantum low-density parity-check code landscape](https://doi.org/10.1103/PhysRevResearch.2.043423), *Phys. Rev. Research* **2**, 043423 (2020). [arXiv:2005.07016](https://arxiv.org/abs/2005.07016).  
   Constant-rate MacKay–Neal HGP family (Table I); semitopological codes and edge-augmentation (Table II, Fig. 2).

2. Joschka Roffe, Lawrence Z. Cohen, Armanda O. Quintavalle, Daryus Chandra, and Earl T. Campbell, [Bias-tailored quantum LDPC codes](https://doi.org/10.22331/q-2023-05-15-1005), *Quantum* **7**, 1005 (2023). [arXiv:2202.01702](https://arxiv.org/abs/2202.01702).  
   Published length-16 seed matrix (Example 3.2, Appendix A) used by `mackay:16`.

3. Jean-Pierre Tillich and Gilles Zémor, [Quantum LDPC codes with positive rate and minimum distance proportional to the square root of the blocklength](https://doi.org/10.1109/TIT.2013.2292061), *IEEE Trans. Inf. Theory* **60**, 1193 (2014). [arXiv:0903.0566](https://arxiv.org/abs/0903.0566).  
   Original hypergraph product construction.

4. David J. C. MacKay and Radford M. Neal, [Near Shannon limit performance of low density parity check codes](https://doi.org/10.1049/el:19961141), *Electron. Lett.* **32**, 1645 (1996).  
   MacKay–Neal construction of the (3,4)-regular classical seeds.

5. Michael A. Perlin, [qLDPC](https://github.com/qLDPCOrg/qLDPC) (2023).  
   Library used to build HGP codes, distances, and logical operators.

### BibTeX

```bibtex
@article{Roffe2020Decoding,
  author  = {Roffe, Joschka and White, David R. and Burton, Simon and Campbell, Earl},
  title   = {Decoding across the quantum low-density parity-check code landscape},
  journal = {Phys. Rev. Research},
  volume  = {2},
  pages   = {043423},
  year    = {2020},
  doi     = {10.1103/PhysRevResearch.2.043423},
  eprint  = {2005.07016},
}

@article{Roffe2023Bias,
  author  = {Roffe, Joschka and Cohen, Lawrence Z. and Quintavalle, Armanda O. and Chandra, Daryus and Campbell, Earl T.},
  title   = {Bias-tailored quantum {LDPC} codes},
  journal = {Quantum},
  volume  = {7},
  pages   = {1005},
  year    = {2023},
  doi     = {10.22331/q-2023-05-15-1005},
  eprint  = {2202.01702},
}

@article{Tillich2014HGP,
  author  = {Tillich, Jean-Pierre and Z{\'e}mor, Gilles},
  title   = {Quantum {LDPC} codes with positive rate and minimum distance proportional to the square root of the blocklength},
  journal = {IEEE Transactions on Information Theory},
  volume  = {60},
  number  = {2},
  pages   = {1193--1202},
  year    = {2014},
  doi     = {10.1109/TIT.2013.2292061},
  eprint  = {0903.0566},
}

@article{MacKay1996Neal,
  author  = {MacKay, David J. C. and Neal, Radford M.},
  title   = {Near {Shannon} limit performance of low density parity check codes},
  journal = {Electronics Letters},
  volume  = {32},
  number  = {18},
  pages   = {1645--1646},
  year    = {1996},
  doi     = {10.1049/el:19961141},
}

@misc{perlin2023qldpc,
  author       = {Perlin, Michael A.},
  title        = {{qLDPC}},
  year         = {2023},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/qLDPCOrg/qLDPC}},
}
```
