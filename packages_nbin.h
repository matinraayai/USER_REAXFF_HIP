#undef PACKAGE
#define PACKAGE "KOKKOS"
#include "KOKKOS/nbin_kokkos.h"
#undef PACKAGE
#define PACKAGE "KOKKOS"
#include "KOKKOS/nbin_ssa_kokkos.h"
#undef PACKAGE
#define PACKAGE "USER-DPD"
#include "USER-DPD/nbin_ssa.h"
#undef PACKAGE
#define PACKAGE "USER-I-DO-WHAT-I-WANT"
#include "USER-I-DO-WHAT-I-WANT/nbin_standard.h"
#undef PACKAGE
#define PACKAGE "USER-INTEL"
#include "USER-INTEL/nbin_intel.h"
