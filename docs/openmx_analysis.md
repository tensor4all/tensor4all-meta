# OpenMX Code Analysis for Rust Migration

**Date: 2026-02-26**

This document summarizes the detailed analysis of OpenMX 3.9 source code for planning a Rust migration using tenferro-rs.

## 1. Code Scale Overview

| Metric | Value |
|--------|-------|
| C source files | 346 |
| Header files | 36 |
| Total C lines | ~406,000 |
| Largest single file | `DFTD3vdW_init.c` (33,758 lines) |
| Executables | 62 |
| Test input files | 156 (.dat files) |

### Large Files (>5000 lines)

| File | Lines | Purpose |
|------|-------|---------|
| `DFTD3vdW_init.c` | 33,758 | DFT-D3 van der Waals initialization |
| `Force.c` | 12,598 | Force calculation |
| `Stress.c` | 11,766 | Stress tensor |
| `NBO_Krylov.c` | 10,118 | Natural Bond Orbital analysis |
| `Krylov.c` | 10,655 | O(N) Krylov subspace method |
| `Generate_Wannier.c` | 10,886 | Wannier function generation |
| `truncation.c` | 11,245 | Matrix truncation for O(N) |
| `Input_std.c` | 6,131 | Input file parsing |
| `Total_Energy.c` | 6,856 | Total energy calculation |

## 2. Modularization Status

**Conclusion: No real modularization exists.**

- All 227 core files include `openmx_common.h`
- 330+ global variables shared across all files
- Function signatures reveal nothing about dependencies
- Any function can read/write any global variable

### Global Variable Types

```c
// Examples from openmx_common.h
double **Gxyz;           // Atomic coordinates
double *****Hks;         // Hamiltonian matrices
double *****DM;          // Density matrices
int atomnum;             // Number of atoms
double *****Coulomb_Array;  // LDA+U Coulomb integrals
// ... 330+ more
```

### 6D Arrays (27 total)

Many high-dimensional arrays exist:
- `Hks`, `DM`, `CntH0`, `HNL`, `iHNL` - Hamiltonian/Density matrix components
- `Coulomb_Array`, `AMF_Array` - LDA+U quantities

## 3. Parallelization Mechanisms

### MPI Usage

| Metric | Value |
|--------|-------|
| Files using MPI | 234 |
| Total MPI calls | 5,934 |

**MPI Communication Pattern:**
```c
MPI_CommWD1[myworld1]  // First-level communicator
MPI_CommWD2[myworld2]  // Second-level communicator
Make_Comm_Worlds()     // Dynamic communicator creation
```

Hierarchical parallelization:
- k-point parallelism (level 1)
- Band parallelism (level 2)

### OpenMP Usage

| Metric | Value |
|--------|-------|
| Files using OpenMP | 75 |
| Top files by `#pragma omp` count | |

| File | OpenMP directives |
|------|------------------|
| `Stress.c` | 41 |
| `Total_Energy.c` | 34 |
| `Force.c` | 34 |
| `Band_DFT_Dosout.c` | 34 |

### ScaLAPACK Usage

| Metric | Value |
|--------|-------|
| Files using ScaLAPACK | 16 |
| Main usage | Large-scale cluster calculations |

**Files with most ScaLAPACK calls:**
- `Cluster_DFT_ScaLAPACK.c` (43 calls)
- `Cluster_DFT_NonCol.c` (23 calls)
- `DFT.c` (8 calls)

### Conclusion on MPI Distributed Arrays

**MPI distributed arrays are NOT essential for tenferro-rs.**

- ScaLAPACK used in only 16 files (4.6%)
- Main path uses non-distributed LAPACK
- GPU version uses single-node cuSOLVER
- MPI parallelism can be handled at application layer with `rsmpi`

## 4. Hand-written Einsum (Tensor Contractions)

**Conclusion: Massive amounts of hand-written tensor contractions exist.**

### BLAS Level-3 Usage

| Metric | Value |
|--------|-------|
| Total GEMM calls | 271 |
| Pattern `C = A * B` | ~150 |
| Pattern `C = A^T * B` | ~80 |
| Pattern `C = A * B^T` | ~30 |

### High-dimensional Loops (Not Using GEMM)

**Example 1: LDA+U Energy (Total_Energy.c:2443-2519)**
```c
// 4-level nested loop - tensor contraction
for(ii=0; ii<(2*l1+1); ii++){
  for(jj=0; jj<(2*l1+1); jj++){
    for(kk=0; kk<(2*l1+1); kk++){
      for(ll=0; ll<(2*l1+1); ll++){
        My_Ehub += Coulomb_Array[ii][jj][kk][ll] 
                 * DM_onsite[ii][kk] * DM_onsite[jj][ll];
      }
    }
  }
}
// Equivalent einsum: "ijkl,ik,jl->"
```

**Example 2: Overlap Matrix Construction (Set_OLP_Kin.c:414-441)**
```c
// 6-level nested loop
for (L0=0; L0<=Lmax; L0++){
  for (Mul0=0; Mul0<Mul_max[L0]; Mul0++){
    for (L1=0; L1<=Lmax; L1++){
      for (Mul1=0; Mul1<Mul_max[L1]; Mul1++){
        for (M0=-L0; M0<=L0; M0++){
          TmpOLP[L0][Mul0][M0][L1][Mul1][M1] 
            += Gaunt(L0,M0,L1,M1,l,m) * SumS0[L0][Mul0][L1][Mul1] * CY;
        }
      }
    }
  }
}
// Equivalent einsum: "LMm,lMm,l,m->LMmLMm"
```

**Example 3: Coulomb Interaction (Coulomb_Interaction.c:765-783)**
```c
// 5-level nested loop with Gaunt coefficients
for (i=0; i<(2*l+1); i++){
  for (j=0; j<(2*l+1); j++){
    for (m=0; m<(2*l+1); m++){
      for (n=0; n<(2*l+1); n++){
        for (kk=0; kk<=2*l; kk+=2){
          Coulomb_Sph += Gaunt_SR(l,kk,l-i,l-m) 
                       * Gaunt_SR(l,kk,l-n,l-j) 
                       * Slater_F[kk];
        }
      }
    }
  }
}
```

### Value of tenferro-rs einsum

| Use Case | Value |
|----------|-------|
| Hamiltonian construction (6D loops) | **High** - code simplification, batching |
| LDA+U (4D loops) | **High** - index management elimination |
| Force calculation (3D loops) | **High** - auto-optimization |
| Coulomb integrals (5D loops) | **High** - explicit transpose/conjugate |

**Einsum is more valuable than raw GEMM wrapper because these contractions cannot be easily expressed as single GEMM calls.**

## 5. Post-processing Tools Analysis

### Executables (62 total)

| Category | Count | Examples |
|----------|-------|----------|
| Main DFT | 1 | `openmx` |
| Post-processing | ~15 | `DosMain`, `bandgnu13`, `bin2txt`, `kSpin`, `polB`, `esp` |
| Tests/Benchmarks | ~10 | `test_mpi`, `test_openmp`, `Bench_MatMul` |
| File conversion | ~10 | `cube2xsf`, `pdb2pao`, `frac2xyz` |
| Utilities | ~10 | `add_gcube`, `diff_gcube`, `rot` |
| NEGF/Transport | ~5 | `TRAN_Distribute_Node`, `check_lead` |

### Post-processing Tool Dependencies

| Tool | Lines | Dependencies | MPI | Migration Difficulty |
|------|-------|--------------|-----|---------------------|
| `bandgnu13` | 466 | None (stdio only) | No | **Very Low** |
| `bin2txt` | 297 | None | No | **Very Low** |
| `cube2xsf` | 302 | None | No | **Very Low** |
| `esp` | 1,250 | `Inputtools`, LAPACK | Minimal | **Low** |
| `DosMain` | 3,290 | `Inputtools` | No | **Low** |
| `polB` | 2,859 | `read_scfout`, LAPACK | Yes (Bcast) | **Low-Medium** |
| `kSpin` | 402 | `read_scfout`, MPI | Yes (Init/Fin) | **Low-Medium** |
| `FermiLoop` | ? | MPI-heavy | Yes | **Medium** |

### Key Insight

**Post-processing tools are ideal for Rust migration:**
- Most do NOT depend on `openmx_common.h`
- `Inputtools.h` is a simple parser - easy to reimplement
- `read_scfout.h` is just data structure definitions
- Independent binaries - can be replaced one at a time

## 6. Test Infrastructure

### Current State

| Test Type | Exists | Details |
|-----------|--------|---------|
| Unit tests | **No** | No `assert`, `EXPECT_*`, etc. |
| Integration tests | Yes | `Runtest.c` |
| Post-processing tests | **No** | Manual visual inspection only |

### Runtest.c Mechanism

```c
// Compares generated output with stored reference
dU = fabs(Utot1 - Utot2);  // Total energy difference
dF = fabs(sum1 - sum2);     // Force sum difference
```

**Test output format:**
```
   1  large_example/5_5_13COb2.dat     diff Utot= 0.000000000014  diff Force= 0.000000000007
   2  large_example/B2C62_Band.dat     diff Utot= 0.000000000001  diff Force= 0.000000056268
```

### Test Suites

| Suite | Input Files | Purpose |
|-------|-------------|---------|
| `input_example` | 156 | Small tests (S) |
| `large_example` | 16 | Large tests (L) |
| `large2_example` | ? | Very large tests (L2) |

### Problems

1. No unit tests - cannot verify individual functions
2. Integration tests only - hard to pinpoint failures
3. Comparison with stored `.out` files - can become stale
4. No intermediate data verification - SCF internals not validated

## 7. SCF Core Code Scale

| Category | Files | Lines | Key Files |
|----------|-------|-------|-----------|
| Main SCF loop | 5 | 36,291 | `DFT.c`, `Total_Energy.c`, `Force.c`, `Stress.c`, `Poisson.c` |
| Hamiltonian/Overlap | 15 | 4,230 | `Hamiltonian_*.c`, `Overlap_*.c` |
| Eigenvalue solvers | 31 | 79,512 | `Krylov.c`, `Divide_Conquer*.c`, `Cluster_DFT*.c`, `Band_DFT*.c` |
| Initialization | 15 | 14,649 | `Set_*.c` |
| Mixing | 8 | 9,485 | `DIIS_Mixing*.c`, `Mixing_*.c` |
| Exchange-correlation | 8 | 2,780 | `XC_*.c`, `Set_XC_*.c` |
| Fourier transforms | 6 | 5,987 | `FT_*.c` |
| Common definitions | 2 | 5,489 | `openmx_common.h/c` |
| **Total SCF Core** | **~90** | **~158,423** | |

**Note:** `Force.c` (12,598 lines) and `Stress.c` (11,766 lines) are candidates for automatic differentiation.

## 8. Partial Rust Migration Strategy

### Why Partial Migration is Difficult

The 330+ global variables create a hard barrier:
```c
// Any C function can access these
double **Gxyz;      // Atomic coordinates
double *****Hks;    // Hamiltonian
double *****DM;     // Density matrix
```

Replacing one function requires passing all relevant global state, effectively rewriting the caller as well.

### Feasible Strategies

| Strategy | Feasibility | Approach |
|----------|-------------|----------|
| Post-processing tools | **Easy** | Independent, no `openmx_common.h` dependency |
| File-based validation | **Easy** | Output intermediate data to HDF5, verify in Rust |
| New Rust implementation | **Recommended** | Clean slate, verify against C reference |
| C FFI replacement | **Hard** | Requires passing all global state |

### Recommended Migration Path

```
Phase 0: Post-processing tools
         bandgnu13, bin2txt, cube2xsf (no dependencies)
         ↓
Phase 1: Instrument C code
         Add HDF5 output for intermediate data
         ↓
Phase 2: Build reference database
         Run test suite, collect DM, Hks, eigenvalues, etc.
         ↓
Phase 3: Implement Rust version
         New implementation, crate-based architecture
         ↓
Phase 4: Cross-validate
         Same inputs to C and Rust, compare outputs
```

## 9. OpenMX Differentiation

### Strengths vs VASP/Quantum ESPRESSO

| Feature | OpenMX | VASP | QE |
|---------|--------|------|-----|
| Basis | PAO (localized) | PAW | Plane waves |
| O(N) methods | ✅ Krylov, DC | ❌ | Partial |
| NEGF transport | ✅ Integrated | ❌ | Separate module |
| Non-collinear | ✅ | ✅ | ✅ |
| Spin-orbit | ✅ | ✅ | ✅ |
| Cost | Free (GPL) | $5000-25000 | Free (GPL) |

### Niche Position

```
VASP:  "Industry standard, reliable, paid"
QE:    "Plane waves, free, large community"
OpenMX: "O(N) + NEGF, free, large-scale systems"
```

**Best use cases:**
- 1000+ atom systems
- Nano-device transport calculations
- Large-scale molecular dynamics

### Rust Migration Benefits

| Current Weakness | Rust Improvement |
|------------------|------------------|
| Code maintainability | Modularization, auto tests |
| GPU support | Experimental → tenferro-rs transparent |
| Python integration | pyO3 easy bindings |
| User extensibility | trait-based plugins |
| CI/CD | cargo test automation |

## 10. Conclusions

### Key Findings

1. **No modularization** - 330+ global variables across all files
2. **MPI distributed arrays not essential** - ScaLAPACK used in only 4.6% of files
3. **Einsum valuable** - Many hand-written high-dimensional contractions
4. **Post-processing tools easy to migrate** - Independent of main codebase
5. **No unit tests** - Integration tests only
6. **Single developer** - Low barrier to architectural changes

### Recommended Priority for tenferro-rs

| Feature | Priority | Reason |
|---------|----------|--------|
| `einsum` | **High** | Many 4-6D contractions not expressible as single GEMM |
| `gemm`/`eigen` | High | Core operations, GPU transparency valuable |
| Batched linalg | Medium | Multiple k-points, spins |
| MPI distributed | Low | Application layer concern |

### Next Steps

1. Start with post-processing tools (bandgnu13, etc.)
2. Add HDF5 instrumentation to C code
3. Build reference test database
4. Design Rust crate architecture
5. Implement and validate incrementally
