"""Deterministic test of the SpineCache growth/shrink state machine with a
synthetic kernel.  The real machine is currently at memorystatus=critical, so
the live path can only be observed refusing to grow; this drives every branch."""
import os, sys
sys.path.insert(0, "/Users/chris/Claude/kimi-k3/tools")
os.environ["K3_SPINE_CACHE_GB"] = "3"
os.environ["K3_SPINE_CACHE_AUTO"] = "1"
os.environ["K3_SPINE_CACHE_FLOOR_GB"] = "8"
os.environ["K3_SPINE_CACHE_TOL_GB"] = "0.5"
os.environ["K3_SPINE_CACHE_SHRINK"] = "2"
import spine_cache as sc

PAGE = sc.PAGE
KERNEL = {"free": 40e9, "purgeable": 0, "speculative": 0, "external": 20e9,
          "compressions": 0.0, "swapouts": 0.0, "level": 1}


def fake_snap():
    return {"free": int(KERNEL["free"] / PAGE), "purgeable": 0, "speculative": 0,
            "external": int(KERNEL["external"] / PAGE),
            "compressions": int(KERNEL["compressions"] / PAGE),
            "swapouts": int(KERNEL["swapouts"] / PAGE)}


sc.vm_snapshot = fake_snap
sc.pressure_level = lambda: KERNEL["level"]

LOG = []


class FakeSF:
    KEEP = set()
    def pin(self, *b):   [self.KEEP.add(id(x)) for x in b if x is not None]
    def unpin(self, *b): [self.KEEP.discard(id(x)) for x in b if x is not None]


sf = FakeSF()
c = sc.SpineCache(sf, n_layers=92, log=lambda m: LOG.append(m) or print("   ", m))
P = lambda i: {"q": bytearray(600_000_000), "sc": bytearray(1000), "other": bytearray(1000)}
NB = 600_002_000
fails = []


def step(label, expect_admit=None, expect_layers=None, i=[0]):
    p = P(i[0])
    ok = c.admit(f"L{i[0]}.", p, NB)
    if not ok:
        c.poll()
    print(f"{label:44s} admit={ok!s:5s} layers={len(c.d)} "
          f"{c.bytes/1e9:4.1f} GB frozen={c.frozen} pinned={len(sf.KEEP)}")
    if expect_admit is not None and ok != expect_admit:
        fails.append(f"{label}: admit={ok} expected {expect_admit}")
    if expect_layers is not None and len(c.d) != expect_layers:
        fails.append(f"{label}: layers={len(c.d)} expected {expect_layers}")
    i[0] += 1


print("1. quiet machine, plenty of headroom -> grows to the 3 GB ceiling")
step("  admit #1", True, 1)
step("  admit #2", True, 2)
step("  admit #3", True, 3)
step("  admit #4", True, 4)
step("  admit #5 (would exceed 3 GB ceiling)", False, 4)
assert c.frozen, "ceiling should latch"
assert len(sf.KEEP) == 12, sf.KEEP

print("\n2. compressor starts moving -> shrink, and the pins are released")
c.frozen = False
KERNEL["compressions"] += 1.2e9
c._poll_n = c.POLL_EVERY - 1
c.poll()
print(f"    after poll: layers={len(c.d)} {c.bytes/1e9:.1f} GB pinned={len(sf.KEEP)}")
if len(c.d) != 2:
    fails.append(f"shrink: expected 2 layers left, got {len(c.d)}")
if len(sf.KEEP) != 6:
    fails.append(f"shrink: expected 6 pinned buffers left, got {len(sf.KEEP)}")

print("\n3. pressure latches growth off even once the compressor calms down")
KERNEL["compressions"] += 0
step("  admit after pressure", False, 2)

print("\n4. swapout -> CRITICAL -> drop everything")
c2 = sc.SpineCache(sf, n_layers=92, log=lambda m: print("   ", m))
KERNEL["compressions"] = 10e9
KERNEL["swapouts"] = 10e9
for k in range(3):
    p = P(k)
    c2.admit(f"M{k}.", p, NB)
print(f"    seeded: {len(c2.d)} layers")
KERNEL["swapouts"] += 2e9
c2._poll_n = c2.POLL_EVERY - 1
c2.poll()
print(f"    after critical poll: layers={len(c2.d)} {c2.bytes/1e9:.1f} GB")
if len(c2.d) != 0:
    fails.append(f"critical: expected full drop, got {len(c2.d)}")

print("\n5. low `free` alone must NOT shrink (healthy page-cache steady state)")
c3 = sc.SpineCache(sf, n_layers=92, log=lambda m: print("   ", m))
KERNEL["free"] = 0.3e9
KERNEL["external"] = 40e9
for k in range(2):
    c3.admit(f"N{k}.", P(k), NB)
c3._poll_n = c3.POLL_EVERY - 1
c3.poll()
print(f"    layers={len(c3.d)} frozen={c3.frozen}  (external 40 GB covers it)")
if len(c3.d) != 2 or c3.frozen:
    fails.append(f"low-free: shrank or froze when it should not have "
                 f"(layers={len(c3.d)} frozen={c3.frozen})")

print("\n6. AUTO growth gate: free AND file cache both exhausted -> pause, no latch")
KERNEL["free"] = 0.3e9
KERNEL["external"] = 2e9
c4 = sc.SpineCache(sf, n_layers=92, log=lambda m: print("   ", m))
ok = c4.admit("Z0.", P(0), NB)
print(f"    admit={ok} frozen={c4.frozen} skipped={c4.skipped}")
if ok or c4.frozen:
    fails.append("headroom pause must not latch frozen")
KERNEL["free"] = 40e9
ok = c4.admit("Z1.", P(1), NB)
print(f"    headroom returns -> admit={ok}")
if not ok:
    fails.append("cache did not resume after headroom returned")

print("\n" + (f"FAIL: {fails}" if fails else "ALL FSM CHECKS PASSED"))
sys.exit(1 if fails else 0)
