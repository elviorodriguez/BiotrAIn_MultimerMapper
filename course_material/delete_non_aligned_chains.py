from chimerax.core.commands import run
from chimerax.geometry import distance

REF_SPEC = "#1/B"
# Max mean Cα distance (Å) to consider a chain as "aligned" to the reference
DIST_CUTOFF = 10.0
# Minimum fraction of reference Cα atoms that must have a nearby chain Cα
COVERAGE_CUTOFF = 0.3

# ── Get reference Cα coordinates ─────────────────────────────────────────────
ref_chain = None
for m in session.models.list():
    if hasattr(m, "chains") and m.id_string == "1":
        for c in m.chains:
            if c.chain_id == "B":
                ref_chain = c
                break

if ref_chain is None:
    raise ValueError("Reference chain #1/B not found.")

ref_cas = [r.find_atom("CA") for r in ref_chain.existing_residues]
ref_cas = [a for a in ref_cas if a is not None]
ref_coords = [a.scene_coord for a in ref_cas]

if not ref_coords:
    raise ValueError("No Cα atoms found in reference chain #1/B.")

session.logger.info(f"Reference #1/B has {len(ref_coords)} Cα atoms.")


# ── Score a chain by geometric proximity to reference Cα atoms ───────────────
def score_chain(chain):
    """
    For each Cα in the candidate chain, find the nearest reference Cα.
    Returns (mean_min_dist, coverage) where:
      - mean_min_dist: average of per-atom minimum distances (lower = better)
      - coverage: fraction of *reference* Cα atoms that have at least one
                  candidate Cα within DIST_CUTOFF (higher = better)
    """
    cas = [r.find_atom("CA") for r in chain.existing_residues]
    cas = [a for a in cas if a is not None]
    if not cas:
        return float("inf"), 0.0

    coords = [a.scene_coord for a in cas]

    # For each candidate Cα, find minimum distance to any reference Cα
    min_dists = []
    for cc in coords:
        d = min(distance(cc, rc) for rc in ref_coords)
        min_dists.append(d)

    mean_dist = sum(min_dists) / len(min_dists)

    # Coverage: how many reference Cα atoms are "covered" by this chain
    covered = 0
    for rc in ref_coords:
        if any(distance(rc, cc) < DIST_CUTOFF for cc in coords):
            covered += 1
    coverage = covered / len(ref_coords)

    return mean_dist, coverage


# ── Main loop ─────────────────────────────────────────────────────────────────
models = [m for m in session.models.list() if hasattr(m, "chains")]

kept    = []
removed = []

for model in models:
    if model.id_string == "1":
        kept.append("#1/B  (reference)")
        continue

    chain_scores = []

    for chain in model.chains:
        spec = f"#{model.id_string}/{chain.chain_id}"
        mean_dist, coverage = score_chain(chain)
        chain_scores.append((chain, mean_dist, coverage, spec))
        session.logger.info(
            f"  {spec}  mean_dist={mean_dist:.2f}Å  coverage={coverage:.2%}"
        )

    if not chain_scores:
        continue

    # Filter chains that meet the cutoffs
    passing = [
        (chain, d, cov, spec)
        for chain, d, cov, spec in chain_scores
        if d < DIST_CUTOFF and cov >= COVERAGE_CUTOFF
    ]

    if not passing:
        # Fallback: just keep the geometrically closest chain
        session.logger.warning(
            f"  No chain in #{model.id_string} passed cutoffs — keeping closest."
        )
        passing = [min(chain_scores, key=lambda x: x[1])]

    # Among passing chains, keep the one with best coverage (tiebreak: distance)
    best = max(passing, key=lambda x: (x[2], -x[1]))
    best_chain, best_dist, best_cov, best_spec = best
    kept.append(f"{best_spec}  mean_dist={best_dist:.2f}Å  coverage={best_cov:.2%}")

    for chain, d, cov, spec in chain_scores:
        if chain is best_chain:
            continue
        try:
            run(session, f"delete {spec}")
            removed.append(f"{spec}  mean_dist={d:.2f}Å  coverage={cov:.2%}")
        except Exception as e:
            removed.append(f"{spec}  ERROR: {e}")

session.logger.info("\n=== KEPT ===\n"    + "\n".join(kept))
session.logger.info("\n=== REMOVED ===\n" + "\n".join(removed))