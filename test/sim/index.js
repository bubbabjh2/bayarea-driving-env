// Lets `node --test test/sim/` (a directory argument) run every sim test file: Node 22
// resolves the directory to this index module, which imports the *.test.js files.
// `node --test test/sim/*.test.js` runs the same tests as separate files.
import './collisions.test.js';
import './physics.test.js';
import './population.test.js';
import './sim.test.js';
import './spatial.test.js';
import './vehicle.test.js';
import './env.test.js';
