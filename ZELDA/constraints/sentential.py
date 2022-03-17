from pysdd.sdd import SddManager
from pathlib import Path
import math

here = Path(__file__).parent

sdd, root = SddManager.from_cnf_file(bytes(here / "constraints.txt"))
sdd.set_prevent_transformation(True)
sdd.auto_gc_and_minimize_off()

a, b, *rest = sdd.vars

wmc = root.wmc(log_mode=True)

# Positive literal weight
wmc.set_literal_weight(a, 0)
# Negative literal weight
wmc.set_literal_weight(-a, 1)

wmc.set_literal_weight(b, 0)
# Negative literal weight
wmc.set_literal_weight(-b, 1)
#wmc.set_literal_weight(b, 0)
#wmc.set_literal_weight(-b, 1)
w = wmc.propagate()

#print(wmc.literal_weight(b))
#print(f"Model count: {int(math.exp(w))}")

# weighted
print(f"Weighted model count: {math.exp(w)}")

with open(here / "sdd.dot", "w") as out:
    print(sdd.dot(), file=out)